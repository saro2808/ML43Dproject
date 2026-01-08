import sys
import json
from pathlib import Path


def load_config(schema_class, path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return schema_class(**data)


def get_base_save_dir(base_cfg, eval_cfg=None):
    if 'google.colab' in sys.modules:
        print("Running on Colab. Paths set to Google Drive.")
        # Optional: Auto-mount drive if not already mounted
        from google.colab import drive
        drive.mount('/content/drive')
        base = Path(base_cfg.paths.colab_results)
    else:
        print("Running on Local Machine. Paths set to project root.")
        # Save in a 'results' folder in your project root
        base = Path(base_cfg.paths.local_results)

    save_dir = base
    if eval_cfg:
        save_dir = base / eval_cfg.inference.mode
        
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir