from __future__ import annotations

from typing import Any

import torch
from monai.networks.schedulers import DDIMScheduler, DDPMScheduler


def build_ddpm_scheduler(diffusion_cfg: dict[str, Any], device: torch.device) -> DDPMScheduler:
    """Build a MONAI DDPM scheduler for training.

    Args:
        diffusion_cfg (dict[str, Any]): Diffusion configuration block.
        device (torch.device): Device for scheduler tensors.

    Returns:
        DDPMScheduler: Configured scheduler instance.
    """
    schedule = str(diffusion_cfg.get("schedule", "scaled_linear_beta"))
    scheduler = DDPMScheduler(
        num_train_timesteps=int(diffusion_cfg.get("num_train_timesteps", 1000)),
        schedule=schedule,
        beta_start=float(diffusion_cfg.get("beta_start", 0.00085)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.012)),
        clip_sample=False,
    )
    return scheduler.to(device)


def build_ddim_scheduler(
    diffusion_cfg: dict[str, Any],
    num_steps: int,
    device: torch.device,
) -> DDIMScheduler:
    """Build a MONAI DDIM scheduler for sampling.

    Args:
        diffusion_cfg (dict[str, Any]): Diffusion configuration block.
        num_steps (int): Number of inference steps.
        device (torch.device): Device for scheduler tensors.

    Returns:
        DDIMScheduler: Configured scheduler instance with timesteps set.
    """
    schedule = str(diffusion_cfg.get("schedule", "scaled_linear_beta"))
    scheduler = DDIMScheduler(
        num_train_timesteps=int(diffusion_cfg.get("num_train_timesteps", 1000)),
        schedule=schedule,
        beta_start=float(diffusion_cfg.get("beta_start", 0.00085)),
        beta_end=float(diffusion_cfg.get("beta_end", 0.012)),
        clip_sample=False,
    )
    scheduler = scheduler.to(device)
    scheduler.set_timesteps(int(num_steps), device=device)
    return scheduler
