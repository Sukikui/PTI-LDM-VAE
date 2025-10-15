import numpy as np
import torch
from monai.utils.type_conversion import convert_to_numpy


def normalize_batch_for_display(tensor: torch.Tensor, low: int = 2, high: int = 98) -> torch.Tensor:
    """Normalize a batch of images [B, C, H, W] to [0, 1] for TensorBoard display.

    Background pixels (values == 0) remain black, and low reconstructed values
    (< 1e-3) are forced to 0.

    Args:
        tensor: Input tensor of shape [B, C, H, W]
        low: Lower percentile for normalization
        high: Upper percentile for normalization

    Returns:
        Normalized tensor in range [0, 1]
    """
    np_img = tensor.detach().cpu().numpy()
    normed = []

    for b in range(np_img.shape[0]):
        normed_channels = []
        for c in range(np_img.shape[1]):
            slice_ = np_img[b, c]
            mask = slice_ != 0
            if np.any(mask):
                pixels = slice_[mask]
                min_val = np.percentile(pixels, low)
                max_val = np.percentile(pixels, high)
                slice_norm = np.zeros_like(slice_)
                slice_norm[mask] = np.clip((pixels - min_val) / (max_val - min_val + 1e-8), 0, 1)
            else:
                slice_norm = np.zeros_like(slice_)
            slice_norm[slice_norm < 1e-3] = 0.0  # suppression du bruit de fond
            normed_channels.append(slice_norm)
        normed.append(np.stack(normed_channels))  # [C, H, W]

    return torch.tensor(np.stack(normed))  # [B, C, H, W]


def normalize_image_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image to uint8 format for visualization.

    Args:
        image: Numpy array image

    Returns:
        Image normalized to uint8 [0, 255]
    """
    draw_img = image.copy()
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 0.1:
        draw_img /= np.amax(draw_img)
    return (255 * draw_img).astype(np.uint8)


def visualize_2d_image(image) -> np.ndarray:
    """Prepare a 2D image for visualization as RGB.

    Args:
        image: Image array of shape (H, W), can be torch tensor or numpy array

    Returns:
        RGB image as numpy array of shape (H, W, 3) with values in [0, 255]
    """
    image = convert_to_numpy(image)
    draw_img = normalize_image_to_uint8(image)
    return np.stack([draw_img, draw_img, draw_img], axis=-1)


def visualize_one_slice_in_3d_image(image, axis: int = 2) -> np.ndarray:
    """Prepare a 2D slice from a 3D image for visualization as RGB.

    Args:
        image: 3D image array of shape (H, W, D), can be torch tensor or numpy array
        axis: Axis along which to take the center slice (0, 1, or 2)

    Returns:
        RGB image as numpy array of shape (H, W, 3) with values in [0, 255]

    Raises:
        ValueError: If axis is not in [0, 1, 2]
    """
    image = convert_to_numpy(image)
    center = image.shape[axis] // 2

    if axis == 0:
        slice_img = image[center, :, :]
    elif axis == 1:
        slice_img = image[:, center, :]
    elif axis == 2:
        slice_img = image[:, :, center]
    else:
        raise ValueError(f"axis should be in [0, 1, 2], got {axis}")

    draw_img = normalize_image_to_uint8(slice_img)
    return np.stack([draw_img, draw_img, draw_img], axis=-1)
