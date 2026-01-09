from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from monai.networks.schedulers import DDPMScheduler

from pti_ldm_vae.models import DiffusionUNet, VAEModel, create_condition_projector
from pti_ldm_vae.utils.cli_common import load_json_config
from pti_ldm_vae.utils.regression_utils import build_regression_model_from_config, load_regression_checkpoint
from pti_ldm_vae.utils.vae_loader import load_vae_config, load_vae_model


def build_frozen_vae(config_file: str, checkpoint: str, device: torch.device) -> tuple[VAEModel, int]:
    """Load and freeze the pretrained VAE.

    Args:
        config_file: Path to VAE config JSON.
        checkpoint: Path to VAE weights.
        device: Target device.

    Returns:
        Tuple of VAE model and latent channel count.
    """
    cfg = load_vae_config(config_file)
    vae = load_vae_model(cfg, checkpoint, device)
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    latent_channels = cfg.autoencoder_def["latent_channels"]
    return vae, latent_channels


def build_frozen_regressor(
    config_file: str, checkpoint: str, device: torch.device, patch_size: tuple[int, int], targets: list[str]
):
    """Load and freeze the regression head (VAE encoder + MLP).

    Args:
        config_file: Path to regression config JSON.
        checkpoint: Path to head checkpoint.
        device: Target device.
        patch_size: Patch size used for inference.
        targets: Ordered list of target names.

    Returns:
        Frozen regression model.
    """
    reg_config = load_json_config(config_file)
    data_cfg = reg_config.get("data", {})
    data_cfg["patch_size"] = list(patch_size)
    reg_config["data"] = data_cfg
    model, _ = build_regression_model_from_config(reg_config, targets, device)
    load_regression_checkpoint(Path(checkpoint), model, targets)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def build_unet(unet_config: dict[str, Any]) -> DiffusionUNet:
    """Instantiate the MONAI diffusion UNet from config.

    Args:
        unet_config: Configuration dictionary.

    Returns:
        DiffusionUNet instance.
    """
    return DiffusionUNet.from_config(unet_config)


def build_condition_modules(latent_channels: int, cross_attention_dim: int):
    """Create condition projectors for spatial latents.

    Args:
        latent_channels: Channels in dentate latent.
        cross_attention_dim: Cross-attention dimension.

    Returns:
        Linear projector.
    """
    return create_condition_projector(latent_channels, cross_attention_dim)


def build_scheduler(schedule_cfg: dict[str, Any]) -> DDPMScheduler:
    """Create a DDPM scheduler from configuration.

    Args:
        schedule_cfg: Scheduler configuration.

    Returns:
        DDPMScheduler instance.
    """
    return DDPMScheduler(
        num_train_timesteps=schedule_cfg.get("num_train_timesteps", 1000),
        schedule="scaled_linear_beta",
        beta_start=schedule_cfg.get("beta_start", 0.00085),
        beta_end=schedule_cfg.get("beta_end", 0.012),
    )
