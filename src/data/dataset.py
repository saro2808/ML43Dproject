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

    def __init__(self, root, scene, tokenizer, negative_prob=0.5, max_samples=1000, seed=0):
        self.root = Path(root)
        self.scene = scene
        self.scene_dir = self.root / scene
        self.views = sorted(self.scene_dir.iterdir())
        self.tokenizer = tokenizer
        self.negative_prob = negative_prob

        # preload instance ids per view
        self.view_instances = {}
        for v in self.views:
            ids = np.load(v / "unique_instances.npy")
            self.view_instances[v.name] = ids

        rng = random.Random(seed)
        candidates = []
        for i in range(len(self.views)):
            for j in range(i + 1, len(self.views)):
                vi, vj = self.views[i], self.views[j]
                common = np.intersect1d(
                    self.view_instances[vi.name],
                    self.view_instances[vj.name],
                )
                for inst_id in common:
                    candidates.append((vi, vj, int(inst_id)))
        
        if len(candidates) > max_samples:
            self.samples = rng.sample(candidates, max_samples)
        else:
            self.samples = candidates

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        while True:
            view_i, view_j, inst_id = self.samples[idx]

            rgb_i, mask_i = load_view(view_i)
            rgb_j, mask_j = load_view(view_j)

            # decide positive vs negative
            is_negative = random.random() < self.negative_prob

            if is_negative:
                # choose a different instance in view_j
                candidates = self.view_instances[view_j.name]
                neg_ids = candidates[candidates != inst_id]
                if len(neg_ids) > 0:
                    inst_j = int(random.choice(neg_ids))
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
        
            # Pick a new index if we hit a bad sample
            idx = random.randint(0, len(self.samples) - 1)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop_i},
                    {"type": "image", "image": crop_j},
                    {
                        "type": "text",
                        "text": (
                            "Do these two image regions belong to the same "
                            "physical object in the 3D scene? Answer yes or no."
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": label,
            },
        ]

        return messages
