from __future__ import annotations

from typing import Any

import torch


def read_noise_init_config(config: dict[str, Any]) -> dict[str, float | str]:
    """Extract noise initialization settings from the LDM config.

    Args:
        config (dict[str, Any]): Full LDM configuration dictionary.

    Returns:
        dict[str, float | str]: Normalized noise initialization configuration.
    """
    sampling_cfg = dict(config.get("noise_init", {}))
    return {
        "init_mode": str(sampling_cfg.get("init_mode", "pure_noise")),
        "noise_top": float(sampling_cfg.get("noise_top", 1.0)),
        "noise_bottom": float(sampling_cfg.get("noise_bottom", 0.0)),
        "noise_exponent": float(sampling_cfg.get("noise_exponent", 1.0)),
        "noise_direction": str(sampling_cfg.get("noise_direction", "vertical")),
        "noise_weight": float(sampling_cfg.get("noise_weight", 1.0)),
    }


def build_gradient_noise_mask(
    height: int,
    width: int,
    *,
    noise_top: float,
    noise_bottom: float,
    noise_exponent: float = 1.0,
    direction: str = "vertical",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a gradient noise mask (vertical or horizontal).

    Args:
        height (int): Spatial height.
        width (int): Spatial width.
        noise_top (float): Noise scale at the top of the image.
        noise_bottom (float): Noise scale at the bottom of the image.
        noise_exponent (float): Exponent to shape the vertical gradient.
        direction (str): ``vertical`` (top->bottom) or ``horizontal`` (left->right).
        device (torch.device | None): Optional torch device.
        dtype (torch.dtype | None): Optional tensor dtype.

    Returns:
        torch.Tensor: Mask of shape [1, 1, H, W] broadcastable to latent tensors.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive.")
    if noise_exponent <= 0:
        raise ValueError("noise_exponent must be positive.")
    if noise_top < 0 or noise_bottom < 0:
        raise ValueError("noise_top and noise_bottom must be non-negative.")

    direction_key = direction.strip().lower()
    if direction_key not in {"vertical", "horizontal"}:
        raise ValueError("noise_direction must be 'vertical' or 'horizontal'.")

    if direction_key == "vertical":
        axis = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
        if noise_exponent != 1.0:
            axis = torch.pow(axis, noise_exponent)
        weights = noise_top * (1.0 - axis) + noise_bottom * axis
        weights = weights.view(1, 1, height, 1).expand(1, 1, height, width)
    else:
        axis = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        if noise_exponent != 1.0:
            axis = torch.pow(axis, noise_exponent)
        weights = noise_top * (1.0 - axis) + noise_bottom * axis
        weights = weights.view(1, 1, 1, width).expand(1, 1, height, width)
    return weights


def create_initial_latent(
    z_cond: torch.Tensor,
    *,
    init_mode: str,
    noise_top: float,
    noise_bottom: float,
    noise_exponent: float,
    noise_direction: str,
    noise_weight: float,
) -> torch.Tensor:
    """Create the initial latent for reverse diffusion.

    Args:
        z_cond (torch.Tensor): Dentate latent tensor [B, C, H, W].
        init_mode (str): Either ``pure_noise`` or ``dentate_noisy``.
        noise_top (float): Noise scale at the top of the image.
        noise_bottom (float): Noise scale at the bottom of the image.
        noise_exponent (float): Exponent to shape the vertical gradient.
        noise_direction (str): ``vertical`` or ``horizontal`` gradient direction.
        noise_weight (float): Global noise multiplier.
    Returns:
        torch.Tensor: Initial latent tensor to start sampling.
    """
    if noise_weight < 0:
        raise ValueError("noise_weight must be non-negative.")
    mode = init_mode.strip().lower()
    if mode in {"pure_noise", "noise"}:
        return noise_weight * torch.randn_like(z_cond)
    if mode in {"dentate_noisy", "noisy_dentate", "dentate"}:
        mask = build_gradient_noise_mask(
            z_cond.shape[2],
            z_cond.shape[3],
            noise_top=noise_top,
            noise_bottom=noise_bottom,
            noise_exponent=noise_exponent,
            direction=noise_direction,
            device=z_cond.device,
            dtype=z_cond.dtype,
        )
        noise = noise_weight * torch.randn_like(z_cond)
        return z_cond + noise * mask
    raise ValueError(f"Unsupported init_mode: {init_mode}. Use 'pure_noise' or 'dentate_noisy'.")
