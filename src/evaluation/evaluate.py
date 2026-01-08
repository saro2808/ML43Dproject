import os
import sys
import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_and_save_metrics
from src.utils.vlm_utils import parse_yes_no, qwen_vl_collate_fn, retry_with_forced_prompt


def load_qwen_model(base_cfg, eval_cfg):
    from unsloth import FastVisionModel

    mode = eval_cfg.inference.mode
    model_path = eval_cfg.inference.checkpoint if mode == "lora" else base_cfg.base_model

    model, tokenizer = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=True,
        device_map={"": 0},
    )

    if mode == "lora":
        # STABILITY MODE (For LoRA on T4)
        # Only revert if we are actually in LoRA mode to keep 'base' fast
        # Fixes Recursion/Logic errors
        if hasattr(model, "_old_generate"):
            model.generate = model._old_generate
        
        # Safe way to find layers to fix the mat1/mat2 error (about fp16 and 32)
        layers = getattr(model.language_model, "layers", None)
        if layers:
            for layer in layers:
                if hasattr(layer.mlp, "_old_forward"):
                    layer.mlp.forward = layer.mlp._old_forward
        
        model = model.to(torch.float16)
    else:
        # FAST MODE (For Base Model)
        # Unsloth's compiled kernels work best with the base model
        print('''If the inference hangs consider adding these two lines:
                os.environ["TORCH_COMPILE_DISABLE"] = "1"
                torch._dynamo.disable()''')
        # # T4 INFERENCE: Avoid the 15-minute hang/crash
        # # uncomment this if on T4 and maybe comment for_inference
        # os.environ["TORCH_COMPILE_DISABLE"] = "1"
        # torch._dynamo.disable()
        FastVisionModel.for_inference(model)

    return model, tokenizer


def evaluate(model, tokenizer, dataloader, device, scene, SAVE_DIR, eval_cfg):

    RESULTS_FILE = SAVE_DIR / scene / "predictions.jsonl"
    METRICS_FILE = SAVE_DIR / scene / "metrics.json"
    
    model.eval()

    # resume support
    start_idx = 0
    num_ambiguous = 0
    num_total = 0
    all_preds = []
    all_gts = []

    if RESULTS_FILE.exists():
        print(f"Loading existing progress for {scene}...")
        last_batch = -1
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                record = json.loads(line)
                all_preds.append(record["prediction"])
                all_gts.append(record["gt"])
                last_batch = record["batch"]
        num_total = len(all_preds)
        start_idx = last_batch + 1
        print(f"Resuming from batch {start_idx}. Loaded {len(all_preds)} previous items.")
        
        if METRICS_FILE.exists():
            try:
                m_data = json.loads(METRICS_FILE.read_text())
                num_ambiguous = m_data.get("num_ambiguous_raw", 0)
                print(f"Restored {num_ambiguous} ambiguous counts.")
            except Exception as e:
                print(f"Warning: Could not parse metrics file for ambiguity: {e}")

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    with open(RESULTS_FILE, "a") as fout:
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            if batch_idx < start_idx:
                continue

            # Unpack the dict returned by qwen_vl_collate_fn
            raw_messages = batch_data["raw_batch"]
            batch_meta = batch_data["meta"]

            # Get only the USER messages for the model to predict
            # raw_batch is a list of [user_msg, assistant_msg]
            inference_messages = [[msg[0]] for msg in batch_data["raw_batch"]]
            # Re-tokenize ONLY the user part for the forward pass
            texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                    for m in inference_messages]
            # Extract images (already in the raw_batch)
            images = [[c["image"] for c in m[0]["content"] if c["type"] == "image"]
                     for m in inference_messages]

            # Move new inputs to device
            inputs = tokenizer(text=texts, images=images, return_tensors="pt", padding=True).to(device)

            # This ensures inputs (like pixel_values) match the model's weight type (Half/Float16)
            inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

            # Generate based on the question ONLY
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=eval_cfg.inference.max_new_tokens,
                    do_sample=eval_cfg.inference.do_sample
                )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i, (raw_output, msg_list) in enumerate(zip(decoded, raw_messages)):

                # Get GT from the dataset, NOT the decoded text
                # Get GT from the second message in the pair (the Assistant's label)
                # msg_list[1] is the assistant role dict from your Dataset class
                gt_label_text = msg_list[1]["content"].lower()
                gt = 1 if "yes" in gt_label_text else 0
                # Parse only the new part of the model output
                pred = parse_yes_no(raw_output)

                if pred == -1:
                    retry_text = retry_with_forced_prompt(model, tokenizer, raw_messages[i])
                    pred = parse_yes_no(retry_text)

                if pred == -1:
                    # force wrong prediction
                    pred = 1 - gt
                    num_ambiguous += 1

                record = {
                    "batch": batch_idx,
                    "item": i,
                    "prediction": pred,
                    "gt": gt,
                    "raw_output": raw_output,
                
                    # metadata
                    "view_i": batch_meta[i]["view_i"],
                    "view_j": batch_meta[i]["view_j"],
                    "inst_i": batch_meta[i]["inst_i"],
                    "inst_j": batch_meta[i]["inst_j"],
                    "is_negative": batch_meta[i]["is_negative"],
                }

                fout.write(json.dumps(record) + "\n")

                num_total += 1
                all_preds.append(pred)
                all_gts.append(gt)

            # periodically save progress + metrics
            if batch_idx % eval_cfg.inference.save_every == 0:
                fout.flush()
                compute_and_save_metrics(all_preds, all_gts, num_ambiguous, num_total, METRICS_FILE)
    
    compute_and_save_metrics(all_preds, all_gts, num_ambiguous, num_total, METRICS_FILE)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from unsloth import FastVisionModel
    from src.data.dataset import ReplicaInstancePairDataset
    from src.configs.schema import BaseConfig, EvalConfig
    from src.utils.setup_utils import load_config, get_base_save_dir

    base_cfg = load_config(BaseConfig, "src/configs/base.json")
    eval_cfg = load_config(EvalConfig, "src/configs/eval.json")
    
    SAVE_DIR = get_base_save_dir(base_cfg, eval_cfg)
    
    model, tokenizer = load_qwen_model(base_cfg, eval_cfg)

    print(f"Running in mode {eval_cfg.inference.mode}")
    scenes = base_cfg.scenes.eval
    # if eval_cfg.inference.mode == "base":
    #     # we evaluate the base model on all the scenes for completeness
    #     scenes += base_cfg.scenes.train

    for scene in scenes:
        print(f"Evaluating on scene {scene}.")
        
        dataset = ReplicaInstancePairDataset(
            root=base_cfg.paths.processed_root,
            scene=scene,
            tokenizer=tokenizer,
            cfg=eval_cfg.dataset
        )
        
        loader = DataLoader(
            dataset,
            batch_size=eval_cfg.inference.batch_size,
            shuffle=False,  # Shuffle=False is better for resume support
            collate_fn=lambda b: qwen_vl_collate_fn(b, tokenizer),
        )
        
        # Use a simple context manager to run the evaluation
        with torch.inference_mode():
            evaluate(model, tokenizer, loader, model.device, scene, SAVE_DIR, eval_cfg)