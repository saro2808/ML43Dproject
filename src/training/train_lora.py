import os
import sys
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, ConcatDataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

from src.data.dataset import ReplicaInstancePairDataset
from src.utils.vlm_utils import qwen_vl_collate_fn, parse_yes_no
from src.configs.schema import BaseConfig, TrainConfig
from src.utils.setup_utils import load_config, get_base_save_dir


# -----------------------
# Main training function
# -----------------------
def main():

    base_cfg = load_config(BaseConfig, "src/configs/base.json")
    train_cfg = load_config(TrainConfig, "src/configs/train.json")
    
    BASE_SAVE_DIR = get_base_save_dir(base_cfg)
    
    # -----------------------
    # Load model
    # -----------------------
    model, tokenizer = FastVisionModel.from_pretrained(
        base_cfg.base_model,
        load_in_4bit=True,
        device_map={"": 0},
    )

    torch._dynamo.config.suppress_errors = True
    FastVisionModel.for_training(model)

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=train_cfg.model.finetune_vision,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=train_cfg.model.lora_r,
        lora_alpha=train_cfg.model.lora_r,
        lora_dropout=train_cfg.model.lora_dropout,
        bias="none",
    )

    # -----------------------
    # Dataset
    # -----------------------
    TRAIN_SCENES = base_cfg.scenes.train

    train_datasets = [
        ReplicaInstancePairDataset(
            root=base_cfg.paths.processed_root,
            scene=scene,
            tokenizer=tokenizer,
            cfg=train_cfg.dataset
        )
        for scene in TRAIN_SCENES
    ]

    train_dataset = ConcatDataset(train_datasets)

    # -----------------------
    # Trainer
    # -----------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=SFTConfig(
            output_dir=str(BASE_SAVE_DIR),
            per_device_train_batch_size=train_cfg.trainer.batch_size,
            gradient_accumulation_steps=train_cfg.trainer.grad_accum,
            learning_rate=train_cfg.trainer.learning_rate,
            num_train_epochs=train_cfg.trainer.epochs,
            fp16=train_cfg.trainer.fp16,
            logging_steps=train_cfg.trainer.logging_steps,
            save_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
        ),
    )

    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # -----------------------
    # Training
    # -----------------------
    trainer_stats = trainer.train()
    
    print("Training finished. Saving final adapter...")

    # Save only the LoRA adapters + config
    model.save_pretrained(BASE_SAVE_DIR / "final_adapter")
    tokenizer.save_pretrained(BASE_SAVE_DIR / "final_adapter")

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


if __name__ == "__main__":
    main()
