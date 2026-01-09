from __future__ import annotations

import numpy as np
import torch


def normalize_batch_for_display(tensor: torch.Tensor, low: int = 2, high: int = 98) -> torch.Tensor:
    """Normalize a batch of images [B, C, H, W] to [0, 1] for display.

    Background pixels (values == 0) remain black, and low reconstructed values
    (< 1e-3) are forced to 0.

    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W].
        low (int): Lower percentile for normalization.
        high (int): Upper percentile for normalization.

    Returns:
        torch.Tensor: Normalized tensor in range [0, 1].
    """
    np_img = tensor.detach().cpu().numpy()
    normed: list[np.ndarray] = []

    for b in range(np_img.shape[0]):
        normed_channels: list[np.ndarray] = []
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
            slice_norm[slice_norm < 1e-3] = 0.0
            normed_channels.append(slice_norm)
        normed.append(np.stack(normed_channels))

    return torch.tensor(np.stack(normed))
