from PIL import Image
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
        if isinstance(img, Image.Image):
            return np.array(img)
        if isinstance(img, torch.Tensor):
            return img.cpu().numpy()
        raise TypeError("img should be either a numpy array, PIL Image or torch tensor.")
    
    imgs = [check_shape(to_numpy(img)) for img in imgs]
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


def remap_mask(mask):
    """
    Maps integer values to their ranks. This function is helpful when visualizing
    instance masks; for example in a mask with unique instance IDs [3, 60000, 60001]
    matplotlib's cmaps would assign almost the same color to the instances 60000 and
    60001, thus making difficult to discriminate them on the image.
    """
    # Find all unique values in the array
    unique_vals = np.unique(mask)
    # Filter out the -1 (background/ignore) value
    labels = unique_vals[unique_vals != -1]
    # np.searchsorted finds the indices where mask elements would fit in 'labels'
    # This effectively maps the k-th smallest label to k
    remapped_indices = np.searchsorted(labels, mask)
    # Preserve the -1 values using np.where
    return np.where(mask == -1, -1, remapped_indices)