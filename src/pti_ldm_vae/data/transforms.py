import torch
import numpy as np
from typing import List, Dict, Any, Tuple


class LocalNormalizeByMask:
    """
    Normalize image intensity excluding background (zero pixels).

    This transform computes mean and standard deviation only on non-zero pixels,
    then normalizes the entire image while keeping background at zero.
    """

    def __call__(self, img: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Apply local normalization by mask.

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
    """
    Dictionary version of LocalNormalizeByMask for MONAI pipelines.

    Applies LocalNormalizeByMask to specified keys in a data dictionary.
    """

    def __init__(self, keys: List[str]):
        """
        Initialize transform.

        Args:
            keys: List of dictionary keys to apply normalization to
        """
        self.keys = keys
        self.norm = LocalNormalizeByMask()

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply normalization to specified keys.

        Args:
            data: Dictionary containing images

        Returns:
            Dictionary with normalized images
        """
        for k in self.keys:
            data[k] = torch.tensor(self.norm(data[k]))
        return data


class ToTuple:
    """
    Convert dictionary to tuple for dataloaders that return tuples.

    Used for LDM dataloaders which return (image, condition_image) tuples.
    """

    def __init__(self, keys: List[str]):
        """
        Initialize transform.

        Args:
            keys: List of dictionary keys in desired output order
        """
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Tuple:
        """
        Convert dictionary to tuple.

        Args:
            data: Dictionary containing data

        Returns:
            Tuple of values in order specified by keys
        """
        return tuple(data[k] for k in self.keys)