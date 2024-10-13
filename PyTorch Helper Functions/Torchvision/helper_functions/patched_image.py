import matplotlib.pyplot as plt
import numpy as np

def plot_image_patches(image_permuted, class_names, label, img_size=224, patch_size=16):
    """
    This function divides an image into patches and displays them in a grid.

    Args:
    - image_permuted: Image as a NumPy array with dimensions (H, W, C).
    - class_names: List of class names.
    - label: The label of the image (index corresponding to the class).
    - img_size: Size of the image (both height and width, assuming a square image).
    - patch_size: Size of each patch (assuming the image is evenly divisible by the patch size).

    The image is divided into patches, and the number of patches is determined by:
    - Number of patches per row and column: img_size // patch_size
    - Total number of patches: (img_size // patch_size) * (img_size // patch_size)
    - Patch size: patch_size pixels x patch_size pixels

    The function creates a grid of subplots, with each subplot displaying one patch of the image.
    """

    assert img_size % patch_size == 0, "Image size must be divisible by patch size"
    
    num_patches = img_size // patch_size

    fig, axs = plt.subplots(nrows=num_patches,
                            ncols=num_patches,
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)
    
    for i, patch_height in enumerate(range(0, img_size, patch_size)):
        for j, patch_width in enumerate(range(0, img_size, patch_size)):
            axs[i, j].imshow(image_permuted[patch_height:patch_height + patch_size,
                                            patch_width:patch_width + patch_size,
                                            :])
            axs[i, j].set_ylabel(f"{i + 1}", rotation="horizontal", horizontalalignment="right", verticalalignment="center")
            axs[i, j].set_xlabel(f"{j + 1}")
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()
    
    fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
    plt.show()