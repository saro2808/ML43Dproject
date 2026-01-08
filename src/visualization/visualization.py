import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_imgs(*imgs, titles=None):

    def check_shape(img):
        if len(img.shape) == 2:
            return img
        if len(img.shape) == 3:
            if img.shape[-1] in [3, 4]:
                return img
            if img.shape[0] == 1:
                return img[0]
            if img.shape[-1] == 1:
                return img[..., 0]
        raise ValueError(f"""Shape mismatch. img is allowed to be only of one of the following
                            shapes: H × W (grayscale), H × W × 3 (RGB) or H × W × 4 (RGBA).
                            Your img has shape {img.shape}.""")
    def to_numpy(img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, torch.Tensor):
            return img.cpu().numpy()
        raise TypeError("img should be either a numpy array or torch tensor.")
    
    imgs = [to_numpy(check_shape(img)) for img in imgs]
    if titles:
        if len(titles) != len(imgs):
            raise ValueError("There should be equally many images and titles.")
    else:
        titles = [f'image {i}' for i in range(len(imgs))]
    
    fig, ax = plt.subplots(1, len(imgs), figsize=(5*len(imgs), 5))

    if len(imgs) == 1:
        ax.imshow(imgs[0])
        ax.axis("off") # hide axis ticks
        ax.set_title(titles[0])
    else:
        for i, img in enumerate(imgs):
            ax[i].imshow(img)
            ax[i].axis("off")
            ax[i].set_title(titles[i])
    
    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()