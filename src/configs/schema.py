from pydantic import BaseModel, Field
from typing import List, Optional

class Paths(BaseModel):
    raw_replica: str = "data/replica"
    processed_root: str = "data/processed"
    colab_results: str = "/content/drive/MyDrive/qwen3vl_eval"
    local_results: str = "./results/qwen3vl_eval"

class CameraConfig(BaseModel):
    height: int = 360
    width: int = 640

class SceneSplit(BaseModel):
    train: List[str]
    eval: List[str]

# --- ROOT CONFIGS ---
class BaseConfig(BaseModel):
    project_name: str
    paths: Paths
    camera: CameraConfig
    scenes: SceneSplit
    base_model: str

class InferenceSettings(BaseModel):
    mode: str = "lora"
    # Note: Using Field(alias=...) allows you to use a cleaner 
    # variable name than the one in the JSON
    checkpoint: str = Field(default="checkpoint-3000", alias="checkpoint_to_eval")
    max_new_tokens: int = 10
    batch_size: int = 2
    save_every: int = 20
    do_sample: bool = False

class DatasetSettings(BaseModel):
    max_samples_per_scene: int = 500
    negative_prob: float = 0.5
    seed: int = 0
    max_pairs_per_instance: int = 10
    prompt: str = "Do these two image regions belong to the same physical object in the 3D scene? Answer yes or no."

# --- ROOT EVAL CONFIG ---
class EvalConfig(BaseModel):
    inference: InferenceSettings
    dataset: DatasetSettings

class LoRAModelSettings(BaseModel):
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    finetune_vision: bool = True

class TrainerSettings(BaseModel):
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 1e-4
    epochs: int = 2
    fp16: bool = True
    logging_steps: int = 20

# --- ROOT TRAIN CONFIG ---
class TrainConfig(BaseModel):
    model: LoRAModelSettings
    dataset: DatasetSettings
    trainer: TrainerSettings