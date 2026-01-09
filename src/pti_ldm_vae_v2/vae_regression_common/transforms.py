from __future__ import annotations

import numpy as np
import torch
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, LoadImage, Resize


class LocalNormalizeByMask:
    """Normalize image intensity excluding background (zero pixels).

    This transform computes mean and standard deviation only on non-zero pixels,
    then normalizes the entire image while keeping background at zero.
    """

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray:
        """Apply local normalization by mask.

        Args:
            img (np.ndarray | torch.Tensor): Input image.

        Returns:
            np.ndarray: Normalized image as float32.
        """
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        mask = img != 0
        pixels = img[mask]
        mean = pixels.mean()
        std = pixels.std() if pixels.std() > 1e-5 else 1.0
        img_norm = (img - mean) / std
        img_norm[~mask] = 0.0
        return img_norm.astype(np.float32)


def build_preprocess_transform(patch_size: tuple[int, int]) -> Compose:
    """Create the shared preprocessing pipeline for VAE and regression head.

    Args:
        patch_size (tuple[int, int]): Target spatial size (height, width).

    Returns:
        Compose: MONAI transform for loading, resizing, and normalizing.
    """
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(patch_size),
            LocalNormalizeByMask(),
            EnsureType(dtype=torch.float32),
        ]
    )
