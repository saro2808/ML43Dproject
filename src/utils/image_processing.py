import numpy as np
from PIL import Image
from pathlib import Path
import random


def load_view(view_dir):
    view_dir = Path(view_dir)
    rgb = Image.open(view_dir / "rgb.png").convert("RGB")
    mask = np.load(view_dir / "instance_mask.npy")
    return rgb, mask


def extract_instance_crop(rgb, instance_mask, instance_id, pad=10):
    mask = (instance_mask == instance_id)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    ymin = max(0, ymin - pad)
    xmin = max(0, xmin - pad)
    ymax = min(instance_mask.shape[0], ymax + pad)
    xmax = min(instance_mask.shape[1], xmax + pad)

    crop = np.array(rgb)[ymin:ymax, xmin:xmax]
    crop_mask = mask[ymin:ymax, xmin:xmax]

    crop[~crop_mask] = 0
    return Image.fromarray(crop)
