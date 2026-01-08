from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.image_processing import load_view, extract_instance_crop


class ReplicaInstancePairDataset(Dataset):
    """
    Returns (messages, label_text) for feeding to VLM.
    """

    def __init__(self, root, scene, tokenizer, cfg):
        self.cfg = cfg
        self.root = Path(root)
        self.scene = scene
        self.scene_dir = self.root / scene
        self.views = sorted(self.scene_dir.iterdir())
        self.tokenizer = tokenizer
        self.negative_prob = cfg.negative_prob

        # preload instance ids per view
        self.view_instances = {}
        for v in self.views:
            ids = np.load(v / "unique_instances.npy")
            self.view_instances[v.name] = ids

        # Build instance -> list of (view_i, view_j) pairs
        self.instance_pairs = {}
        for i in range(len(self.views)):
            for j in range(i + 1, len(self.views)):
                vi, vj = self.views[i], self.views[j]
                common = np.intersect1d(
                    self.view_instances[vi.name],
                    self.view_instances[vj.name],
                )
                for inst_id in common:
                    inst_id = int(inst_id)
                    self.instance_pairs.setdefault(inst_id, []).append((vi, vj))

        self.rng = random.Random(cfg.seed)
        self.instances = list(self.instance_pairs.keys())
        # cap total samples but keep instance-uniformity
        self.max_samples = cfg.max_samples_per_scene

        MAX_PAIRS_PER_INSTANCE = cfg.max_pairs_per_instance
        for inst_id in self.instance_pairs:
            pairs = self.instance_pairs[inst_id]
            if len(pairs) > MAX_PAIRS_PER_INSTANCE:
                self.instance_pairs[inst_id] = self.rng.sample(pairs, MAX_PAIRS_PER_INSTANCE)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        while True:
            # sample instance uniformly
            inst_id = self.rng.choice(self.instances)
            # sample a valid view pair for that instance
            view_i, view_j = self.rng.choice(self.instance_pairs[inst_id])

            rgb_i, mask_i = load_view(view_i)
            rgb_j, mask_j = load_view(view_j)

            # decide positive vs negative
            is_negative = self.rng.random() < self.negative_prob

            if is_negative:
                # choose a different instance in view_j
                candidates = self.view_instances[view_j.name]
                neg_ids = candidates[candidates != inst_id]
                if len(neg_ids) > 0:
                    inst_j = int(self.rng.choice(neg_ids))
                    label = "No"
                else:
                    inst_j = inst_id
                    label = "Yes"
            else:
                inst_j = inst_id
                label = "Yes"

            crop_i = extract_instance_crop(rgb_i, mask_i, inst_id)
            crop_j = extract_instance_crop(rgb_j, mask_j, inst_j)

            if crop_i is not None and crop_j is not None:
                break

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop_i},
                    {"type": "image", "image": crop_j},
                    {
                        "type": "text",
                        "text": self.cfg.prompt,
                    },
                ],
                "metadata": {},
            },
            {
                "role": "assistant",
                "content": label,
                "metadata": {},
            },
        ]


        meta = {
            "view_i": view_i.name,
            "view_j": view_j.name,
            "inst_i": inst_id,
            "inst_j": inst_j,
            "is_negative": is_negative,
        }
        
        return {
            "messages": messages,
            "meta": meta,
        }

