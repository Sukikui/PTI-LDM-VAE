from typing import Any

import numpy as np
import torch
import tifffile


class LocalNormalizeByMask:
    """Normalize image intensity excluding background (zero pixels).

    This transform computes mean and standard deviation only on non-zero pixels, then normalizes the entire image while
    keeping background at zero.
    """

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray:
        """Apply local normalization by mask.

        Args:
            img: Input image (numpy array or torch tensor)

        Returns:
            Normalized image as numpy array
        """
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        mask = img != 0
        pixels = img[mask]
        mean = pixels.mean()
        std = pixels.std() if pixels.std() > 1e-5 else 1.0
        img_norm = (img - mean) / std
        img_norm[~mask] = 0.0
        return img_norm.astype(np.float32)


class ApplyLocalNormd:
    """Dictionary version of LocalNormalizeByMask for MONAI pipelines.

    Applies LocalNormalizeByMask to specified keys in a data dictionary.
    """

    def __init__(self, keys: list[str]) -> None:
        """Initialize transform.

        Args:
            keys: List of dictionary keys to apply normalization to
        """
        self.keys = keys
        self.norm = LocalNormalizeByMask()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply normalization to specified keys.

        Args:
            data: Dictionary containing images

        Returns:
            Dictionary with normalized images
        """
        for k in self.keys:
            data[k] = torch.tensor(self.norm(data[k]))
        return data


class TifReader:
    """Callable transform to read TIF files with tifffile."""

    def __call__(self, path: str) -> np.ndarray:
        """Load a TIF file and return a float32 numpy array.

        Args:
            path: Path to the TIF image.

        Returns:
            Image data as float32 numpy array.
        """
        img = tifffile.imread(path)
        return img.astype(np.float32)
