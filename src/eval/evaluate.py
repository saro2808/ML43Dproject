import os
import sys
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.eval.metrics import compute_and_save_metrics
from src.utils.vlm_utils import parse_yes_no, qwen_vl_collate_fn, retry_with_forced_prompt


def get_base_save_dir():
    if 'google.colab' in sys.modules:
        print("Running on Colab. Paths set to Google Drive.")
        # Optional: Auto-mount drive if not already mounted
        from google.colab import drive
        drive.mount('/content/drive')
        return Path("/content/drive/MyDrive/qwen3vl_eval")
    else:
        print("Running on Local Machine. Paths set to project root.")
        # Save in a 'results' folder in your project root
        return Path("./results/qwen3vl_eval")


SAVE_DIR = get_base_save_dir()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = lambda scene: SAVE_DIR / scene / "predictions.jsonl"
METRICS_FILE = lambda scene: SAVE_DIR / scene / "metrics.json"
PROGRESS_FILE = lambda scene: SAVE_DIR / scene / "progress.txt"


def evaluate(model, tokenizer, dataloader, device, scene, save_every=20):
    
    model.eval()

    # resume support
    start_idx = 0
    num_ambiguous = 0
    num_total = 0
    if PROGRESS_FILE(scene).exists():
        start_idx = int(PROGRESS_FILE(scene).read_text().strip())
        print(f"Resuming from batch {start_idx}")

    all_preds = []
    all_gts = []

    os.makedirs(os.path.dirname(RESULTS_FILE(scene)), exist_ok=True)
    
    with open(RESULTS_FILE(scene), "a+") as fout:
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            if batch_idx < start_idx:
                continue

            # Unpack the dict returned by qwen_vl_collate_fn
            raw_messages = batch_data["raw_batch"]

            # 1. FIX: Get only the USER messages for the model to predict
            # raw_batch is a list of [user_msg, assistant_msg]
            inference_messages = [[msg[0]] for msg in batch_data["raw_batch"]]
            # 2. Re-tokenize ONLY the user part for the forward pass
            texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in inference_messages]
            # Extract images (already in the raw_batch)
            images = [[c["image"] for c in m[0]["content"] if c["type"] == "image"]
                     for m in inference_messages]

            # Move new inputs to device
            inputs = tokenizer(text=texts, images=images, return_tensors="pt", padding=True).to(device)

            # 3. Generate based on the question ONLY
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i, (raw_output, msg_list) in enumerate(zip(decoded, raw_messages)):

                # 1. Get GT from the dataset, NOT the decoded text
                # FIX GT: Get it from the second message in the pair (the Assistant's label)
                # msg_list[1] is the assistant role dict from your Dataset class
                gt_label_text = msg_list[1]["content"].lower()
                gt = 1 if "yes" in gt_label_text else 0

                # 2. Parse only the new part of the model output
                pred = parse_yes_no(raw_output)

                # print(f"Item {i} | Real GT: {gt} | Parser Saw: {pred}")

                record = {
                    "batch": batch_idx,
                    "item": i,
                    "prediction": pred,
                    "gt": gt,
                    "raw_output": raw_output,
                }
                fout.write(json.dumps(record) + "\n")

                if pred == -1:
                    retry_text = retry_with_forced_prompt(model, tokenizer, raw_messages[i])
                    pred = parse_yes_no(retry_text)

                if pred == -1:
                    # force wrong prediction
                    pred = 1 - gt
                    num_ambiguous += 1

                num_total += 1

                all_preds.append(pred)
                all_gts.append(gt)

            # periodically save progress + metrics
            if batch_idx % save_every == 0:
                PROGRESS_FILE(scene).write_text(str(batch_idx))
                compute_and_save_metrics(all_preds, all_gts, num_ambiguous, num_total, METRICS_FILE(scene))

    compute_and_save_metrics(all_preds, all_gts, num_ambiguous, num_total, METRICS_FILE(scene))


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from unsloth import FastVisionModel
    from src.data.dataset import ReplicaInstancePairDataset

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name="unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit",
        load_in_4bit=True,
        device_map="auto"
    )

    # to suppress the conflict between Unsloth's compiled kernels and bitsandbytes
    # 4-bit quantization in Google Colab
    # Disable torch compile for the vision components to stop recursion
    torch._dynamo.config.suppress_errors = True
    
    # # CRITICAL: Bypass the Unsloth optimized generate function
    # # which is often the source of the RecursionError in 4-bit VL models
    # if hasattr(model, "_old_generate"):
    #     model.generate = model._old_generate
    # else:
    #     # If using standard transformers generate
    #     model.generate = model.generate

    scenes = ["office0", "office1", "office2", "office3", "office4",
              "room0", "room1", "room2"]
    for scene in scenes:
        print(f"Evaluating on scene {scene}.")
        
        dataset = ReplicaInstancePairDataset(
            root="data/processed",
            scene=scene,
            tokenizer=tokenizer,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,  # Shuffle=False is better for resume support
            collate_fn=lambda b: qwen_vl_collate_fn(b, tokenizer),
        )
        
        # Use a simple context manager to run the evaluation
        with torch.inference_mode():
            evaluate(model, tokenizer, loader, model.device, scene)