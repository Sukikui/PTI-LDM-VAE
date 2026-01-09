from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from pti_ldm_vae_v2.regression_head.checkpoint import load_regression_checkpoint
from pti_ldm_vae_v2.regression_head.config import build_regression_model_from_config, load_config as load_reg_config
from pti_ldm_vae_v2.vae.config import load_config as load_vae_config
from pti_ldm_vae_v2.vae.config import load_model as load_vae_model
from pti_ldm_vae_v2.vae_regression_common import VAELatentRegressor, VAEModel

from .unet import DiffusionUNet


def build_frozen_vae(config_file: str, checkpoint: str, device: torch.device) -> tuple[VAEModel, int]:
    """Load and freeze the pretrained VAE.

    Args:
        config_file (str): Path to VAE config JSON.
        checkpoint (str): Path to VAE weights.
        device (torch.device): Target device.

    Returns:
        tuple[VAEModel, int]: Frozen VAE model and latent channel count.
    """
    cfg = load_vae_config(config_file)
    vae = load_vae_model(cfg, checkpoint, device)
    for param in vae.parameters():
        param.requires_grad = False
    vae.eval()
    latent_channels = cfg.autoencoder_def.get("latent_channels") if hasattr(cfg, "autoencoder_def") else None
    if latent_channels is None:
        latent_channels = getattr(cfg, "latent_channels", None)
    if latent_channels is None:
        raise ValueError("Unable to infer latent_channels from VAE config.")
    return vae, int(latent_channels)


def build_frozen_regressor(
    config_file: str,
    checkpoint: str,
    device: torch.device,
    patch_size: tuple[int, int],
    targets: list[str],
) -> VAELatentRegressor:
    """Load and freeze the regression head (VAE encoder + MLP).

    Args:
        config_file (str): Path to regression config JSON.
        checkpoint (str): Path to head checkpoint.
        device (torch.device): Target device.
        patch_size (tuple[int, int]): Patch size used for inference.
        targets (list[str]): Ordered list of target names.

    Returns:
        VAELatentRegressor: Frozen regression model.
    """
    reg_config = load_reg_config(config_file)
    data_cfg = dict(reg_config.get("data", {}))
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
        unet_config (dict[str, Any]): Configuration dictionary.

    Returns:
        DiffusionUNet: UNet instance.
    """
    return DiffusionUNet.from_config(unet_config)
