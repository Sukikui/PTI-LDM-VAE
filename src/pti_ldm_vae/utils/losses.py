"""Utility helpers for loss computations."""

from __future__ import annotations

import torch


def ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor with three channels by repeating single-channel inputs.

    Args:
        tensor (torch.Tensor): Tensor of shape [B, C, H, W] passed to perceptual loss.

    Returns:
        torch.Tensor: Tensor with three channels suitable for ImageNet backbones.

    Raises:
        ValueError: If ``tensor`` is not four-dimensional or has an unsupported channel count.
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {tensor.shape}")

    channels = tensor.shape[1]
    if channels == 3:
        return tensor
    if channels == 1:
        return tensor.repeat(1, 3, 1, 1)
    raise ValueError(f"Perceptual loss expects 1 or 3 channels, got {channels}")
