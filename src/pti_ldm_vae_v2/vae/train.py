from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from tqdm import tqdm

from .config import apply_overrides, load_config, resolve_run_dirs
from .data import create_train_val_dataloaders
from .losses import (
    compute_ar_vae_loss,
    compute_kl_loss,
    compute_total_loss,
    ensure_three_channels,
    select_intensity_loss,
)
from pti_ldm_vae_v2.vae_regression_common import VAEModel
from pti_ldm_vae_v2.vae_regression_common import init_device_and_seed
from .visualization import normalize_batch_for_display
from .wandb import init_wandb

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VAE training.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Train a VAE for image reconstruction.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to VAE JSON config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs from config.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    return parser.parse_args()


def _coerce_bool(value: Any) -> bool:
    """Convert a value to bool without string pitfalls.

    Args:
        value (Any): Value to convert.

    Returns:
        bool: Converted boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
        return False
    if value is None:
        return False
    return bool(value)


def _prepare_batch(
    batch: torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]],
    device: torch.device,
    ar_vae_enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Move a batch to the target device and unpack attributes if present.

    Args:
        batch (torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]): Batch from dataloader.
        device (torch.device): Target device.
        ar_vae_enabled (bool): Whether AR-VAE is enabled.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor] | None]: Images and optional attributes.
    """
    if isinstance(batch, tuple):
        images, attributes = batch
    else:
        images = batch
        attributes = None

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Unsupported batch type: {type(images)}")

    images = images.to(device)
    if attributes is not None:
        attributes = {k: v.to(device) for k, v in attributes.items()}
    elif ar_vae_enabled:
        raise ValueError("AR-VAE is enabled but attributes are missing from the batch.")

    return images, attributes


@dataclass
class GeneratorOutputs:
    """Container for autoencoder outputs and generator-side losses.

    Attributes:
        reconstruction (torch.Tensor): Reconstructed images.
        z_mu (torch.Tensor): Latent mean tensor.
        z_logvar (torch.Tensor): Latent log-variance tensor.
        recons_loss (torch.Tensor): Reconstruction loss value.
        kl_loss (torch.Tensor): KL divergence loss value.
        perceptual_loss (torch.Tensor): Perceptual loss value.
        adv_gen_loss (torch.Tensor): Adversarial generator loss value.
        ar_loss (torch.Tensor): AR-VAE loss value.
        ar_losses_per_attr (dict[str, torch.Tensor]): Per-attribute AR losses.
    """

    reconstruction: torch.Tensor
    z_mu: torch.Tensor
    z_logvar: torch.Tensor
    recons_loss: torch.Tensor
    kl_loss: torch.Tensor
    perceptual_loss: torch.Tensor
    adv_gen_loss: torch.Tensor
    ar_loss: torch.Tensor
    ar_losses_per_attr: dict[str, torch.Tensor]


def _resolve_ar_config(
    regularized_attributes: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Extract AR-VAE mapping and global delta configuration.

    Args:
        regularized_attributes (dict[str, Any] | None): AR-VAE configuration block.

    Returns:
        tuple[dict[str, dict[str, float]], dict[str, Any]]: Attribute mapping and delta configuration.
    """
    attribute_latent_mapping, delta_global = _resolve_ar_config(regularized_attributes)
    return attribute_latent_mapping, delta_global


def _compute_adv_gen_loss(
    reconstruction: torch.Tensor,
    discriminator: torch.nn.Module | None,
    adv_loss: PatchAdversarialLoss | None,
    *,
    adv_enabled: bool,
    epoch: int,
) -> torch.Tensor:
    """Compute adversarial generator loss when enabled.

    Args:
        reconstruction (torch.Tensor): Reconstructed images.
        discriminator (torch.nn.Module | None): Discriminator network.
        adv_loss (PatchAdversarialLoss | None): Adversarial loss helper.
        adv_enabled (bool): Whether adversarial training is enabled.
        epoch (int): Current epoch index.

    Returns:
        torch.Tensor: Adversarial generator loss.
    """
    if _adv_is_active(
        adv_enabled=adv_enabled,
        adv_loss=adv_loss,
        discriminator=discriminator,
        epoch=epoch,
    ):
        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
        return adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
    return torch.tensor(0.0, device=reconstruction.device)


def _compute_adv_disc_loss(
    reconstruction: torch.Tensor,
    images: torch.Tensor,
    discriminator: torch.nn.Module | None,
    adv_loss: PatchAdversarialLoss | None,
    *,
    adv_enabled: bool,
    epoch: int,
) -> torch.Tensor:
    """Compute adversarial discriminator loss when enabled.

    Args:
        reconstruction (torch.Tensor): Reconstructed images.
        images (torch.Tensor): Input images.
        discriminator (torch.nn.Module | None): Discriminator network.
        adv_loss (PatchAdversarialLoss | None): Adversarial loss helper.
        adv_enabled (bool): Whether adversarial training is enabled.
        epoch (int): Current epoch index.

    Returns:
        torch.Tensor: Adversarial discriminator loss.
    """
    if _adv_is_active(
        adv_enabled=adv_enabled,
        adv_loss=adv_loss,
        discriminator=discriminator,
        epoch=epoch,
    ):
        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        return (loss_d_fake + loss_d_real) * 0.5
    return torch.tensor(0.0, device=reconstruction.device)


def _compute_generator_outputs(
    *,
    images: torch.Tensor,
    autoencoder: torch.nn.Module,
    intensity_loss: torch.nn.Module,
    loss_perceptual: PerceptualLoss,
    discriminator: torch.nn.Module | None,
    adv_loss: PatchAdversarialLoss | None,
    adv_enabled: bool,
    epoch: int,
    ar_vae_enabled: bool,
    batch_attributes: dict[str, torch.Tensor] | None,
    attribute_latent_mapping: dict[str, dict[str, float]],
    pairwise_mode: str,
    subset_pairs: int | None,
    delta_global: dict[str, Any],
    cast_recons_to_float: bool,
    adv_active: bool,
) -> GeneratorOutputs:
    """Run the autoencoder forward pass and compute generator-side losses.

    Args:
        images (torch.Tensor): Input images.
        autoencoder (torch.nn.Module): VAE model.
        intensity_loss (torch.nn.Module): Intensity reconstruction loss.
        loss_perceptual (PerceptualLoss): Perceptual loss module.
        discriminator (torch.nn.Module | None): Discriminator network.
        adv_loss (PatchAdversarialLoss | None): Adversarial loss helper.
        adv_enabled (bool): Whether adversarial training is enabled.
        epoch (int): Current epoch index.
        ar_vae_enabled (bool): Whether AR-VAE is enabled.
        batch_attributes (dict[str, torch.Tensor] | None): Attributes for AR-VAE.
        attribute_latent_mapping (dict[str, dict[str, float]]): Attribute-to-latent mapping.
        pairwise_mode (str): Pairwise mode for AR-VAE.
        subset_pairs (int | None): Subset size for AR-VAE pairs.
        delta_global (dict[str, Any]): Global delta configuration for AR-VAE.
        cast_recons_to_float (bool): Whether to cast recon/images to float for intensity loss.
        adv_active (bool): Whether adversarial losses are active.

    Returns:
        GeneratorOutputs: Reconstruction outputs and generator losses.
    """
    reconstruction, z_mu, z_logvar = autoencoder(images)
    if cast_recons_to_float:
        recons_loss = intensity_loss(reconstruction.float(), images.float())
    else:
        recons_loss = intensity_loss(reconstruction, images)
    kl_loss = compute_kl_loss(z_mu, z_logvar)
    perceptual_loss = loss_perceptual(
        ensure_three_channels(reconstruction.float()),
        ensure_three_channels(images.float()),
    )

    adv_gen_loss = torch.tensor(0.0, device=images.device)
    if adv_active:
        adv_gen_loss = _compute_adv_gen_loss(
            reconstruction,
            discriminator,
            adv_loss,
            adv_enabled=adv_enabled,
            epoch=epoch,
        )

    ar_loss = torch.tensor(0.0, device=images.device)
    ar_losses_per_attr: dict[str, torch.Tensor] = {}
    if ar_vae_enabled:
        ar_loss, ar_losses_per_attr, _, _ = compute_ar_vae_loss(
            latent_vectors=z_mu,
            attributes=batch_attributes if batch_attributes is not None else {},
            attribute_latent_mapping=attribute_latent_mapping,
            pairwise_mode=pairwise_mode,
            subset_pairs=subset_pairs,
            delta_global=delta_global,
        )

    return GeneratorOutputs(
        reconstruction=reconstruction,
        z_mu=z_mu,
        z_logvar=z_logvar,
        recons_loss=recons_loss,
        kl_loss=kl_loss,
        perceptual_loss=perceptual_loss,
        adv_gen_loss=adv_gen_loss,
        ar_loss=ar_loss,
        ar_losses_per_attr=ar_losses_per_attr,
    )


def _adv_is_active(
    *,
    adv_enabled: bool,
    adv_loss: PatchAdversarialLoss | None,
    discriminator: torch.nn.Module | None,
    epoch: int,
) -> bool:
    """Check whether adversarial loss should be applied.

    Args:
        adv_enabled (bool): Whether adversarial training is enabled.
        adv_loss (PatchAdversarialLoss | None): Adversarial loss helper.
        discriminator (torch.nn.Module | None): Discriminator network.
        epoch (int): Current epoch index.

    Returns:
        bool: True when adversarial losses should be computed.
    """
    return adv_enabled and adv_loss is not None and discriminator is not None and epoch > 5


def _build_models(config: Any, device: torch.device, adv_enabled: bool) -> tuple[VAEModel, torch.nn.Module | None]:
    """Instantiate the VAE and optional discriminator.

    Args:
        config (Any): Parsed config with ``autoencoder_def`` and ``spatial_dims``.
        device (torch.device): Target device.
        adv_enabled (bool): Whether to build the discriminator.

    Returns:
        tuple[VAEModel, torch.nn.Module | None]: Autoencoder and optional discriminator.
    """
    autoencoder = VAEModel.from_config(config.autoencoder_def).to(device)

    discriminator = None
    if adv_enabled:
        discriminator = PatchDiscriminator(
            spatial_dims=config.spatial_dims,
            num_layers_d=3,
            channels=32,
            in_channels=1,
            out_channels=1,
            norm="INSTANCE",
        ).to(device)

    return autoencoder, discriminator


def _build_losses(
    train_cfg: dict[str, Any],
    device: torch.device,
    spatial_dims: int,
    adv_enabled: bool,
) -> tuple[torch.nn.Module, PatchAdversarialLoss | None, PerceptualLoss]:
    """Create loss functions used during training.

    Args:
        train_cfg (dict[str, Any]): Training configuration.
        device (torch.device): Target device.
        spatial_dims (int): Number of spatial dimensions for perceptual loss.
        adv_enabled (bool): Whether to build adversarial loss.

    Returns:
        tuple[torch.nn.Module, PatchAdversarialLoss | None, PerceptualLoss]: Intensity, adversarial, and perceptual loss.
    """
    intensity_loss = select_intensity_loss(train_cfg.get("recon_loss"), verbose=True)

    adv_loss = PatchAdversarialLoss(criterion="least_squares") if adv_enabled else None
    loss_perceptual = PerceptualLoss(spatial_dims=spatial_dims, network_type="squeeze").to(device)
    return intensity_loss, adv_loss, loss_perceptual


def _build_optimizers(
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    lr: float,
    adv_enabled: bool,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]:
    """Create optimizers for the generator and discriminator.

    Args:
        autoencoder (torch.nn.Module): VAE model.
        discriminator (torch.nn.Module | None): Optional discriminator.
        lr (float): Learning rate.
        adv_enabled (bool): Whether adversarial training is enabled.

    Returns:
        tuple[torch.optim.Optimizer, torch.optim.Optimizer | None]: Generator and discriminator optimizers.
    """
    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr) if adv_enabled and discriminator else None
    return optimizer_g, optimizer_d


def _load_checkpoint(
    config: Any,
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[int, float, int, int | None]:
    """Load training state from a checkpoint if requested.

    Args:
        config (Any): Parsed config containing resume settings.
        autoencoder (torch.nn.Module): VAE model.
        discriminator (torch.nn.Module | None): Optional discriminator.
        optimizer_g (torch.optim.Optimizer): Generator optimizer.
        optimizer_d (torch.optim.Optimizer | None): Discriminator optimizer.
        device (torch.device): Target device.

    Returns:
        tuple[int, float, int, int | None]: start_epoch, best_val_loss, total_step, best_epoch_saved.
    """
    if not getattr(config, "resume_ckpt", False):
        print("[INFO] Training from scratch")
        return 0, 100.0, 0, None

    checkpoint_path = getattr(config, "checkpoint_dir", "")
    if not checkpoint_path:
        raise ValueError("resume_ckpt is true but checkpoint_dir is empty.")

    print(f"[INFO] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    if discriminator is not None and checkpoint.get("discriminator_state_dict") is not None:
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
    if optimizer_d is not None and checkpoint.get("optimizer_d_state_dict") is not None:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint["best_val_loss"])
    total_step = int(checkpoint["total_step"])
    best_epoch_saved = int(checkpoint["epoch"])
    print(f"[INFO] Resuming from epoch {start_epoch} | best_val_loss = {best_val_loss:.4f}")
    return start_epoch, best_val_loss, total_step, best_epoch_saved


def _log_triplet(
    wandb_run: Any,
    *,
    tag: str,
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    step: int | None = None,
    caption: str | None = None,
) -> None:
    """Log a triplet (original | reconstruction | diff) to W&B.

    Args:
        wandb_run (Any): Active W&B run.
        tag (str): W&B key for the image list.
        image (torch.Tensor): Input image tensor [1, H, W].
        reconstruction (torch.Tensor): Reconstruction tensor [1, H, W].
        step (int | None): Optional W&B step.
        caption (str | None): Optional caption.
    """
    try:
        import wandb  # type: ignore
    except ImportError:
        return

    diff = torch.abs(image - reconstruction)
    img_disp = torch.rot90(normalize_batch_for_display(image.unsqueeze(0)), k=3, dims=[2, 3])[0]
    recon_disp = torch.rot90(normalize_batch_for_display(reconstruction.unsqueeze(0)), k=3, dims=[2, 3])[0]
    diff_disp = torch.rot90(normalize_batch_for_display(diff.unsqueeze(0)), k=3, dims=[2, 3])[0]

    triplet = torch.cat([img_disp, recon_disp, diff_disp], dim=2)
    payload = {tag: [wandb.Image(triplet.permute(1, 2, 0).numpy(), caption=caption)]}
    if step is None:
        wandb_run.log(payload)
    else:
        wandb_run.log(payload, step=step)


def _save_split(
    splits_dir: Path,
    *,
    seed: int,
    train_split: float,
    val_dir: str | None,
    train_paths: list[str],
    val_paths: list[str],
) -> None:
    """Save train/val split metadata.

    Args:
        splits_dir (Path): Directory to write the split JSON.
        seed (int): Random seed used for splitting.
        train_split (float): Train/val ratio.
        val_dir (str | None): Optional validation directory.
        train_paths (list[str]): Training file paths.
        val_paths (list[str]): Validation file paths.
    """
    split_payload = {
        "seed": seed,
        "train_split": train_split,
        "val_dir": val_dir,
        "train_files": list(train_paths),
        "val_files": list(val_paths),
    }
    split_path = splits_dir / "vae_split.json"
    with open(split_path, "w", encoding="utf-8") as split_file:
        json.dump(split_payload, split_file, indent=2)
    print(f"[INFO] Saved train/val split to {split_path}")


def _train_epoch(
    *,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer | None,
    intensity_loss: torch.nn.Module,
    adv_loss: PatchAdversarialLoss | None,
    loss_perceptual: PerceptualLoss,
    kl_weight: float,
    perceptual_weight: float,
    adv_weight: float,
    ar_gamma: float,
    device: torch.device,
    total_step: int,
    wandb_run: Any,
    ar_vae_enabled: bool,
    regularized_attributes: dict[str, Any] | None,
    pairwise_mode: str,
    subset_pairs: int | None,
    adv_enabled: bool,
    max_epochs: int,
) -> int:
    """Run one training epoch.

    Args:
        epoch (int): Current epoch index.
        train_loader (torch.utils.data.DataLoader): Training loader.
        autoencoder (torch.nn.Module): VAE model.
        discriminator (torch.nn.Module | None): Optional discriminator.
        optimizer_g (torch.optim.Optimizer): Generator optimizer.
        optimizer_d (torch.optim.Optimizer | None): Discriminator optimizer.
        intensity_loss (torch.nn.Module): Intensity loss.
        adv_loss (PatchAdversarialLoss | None): Adversarial loss.
        loss_perceptual (PerceptualLoss): Perceptual loss.
        kl_weight (float): KL weight.
        perceptual_weight (float): Perceptual loss weight.
        adv_weight (float): Adversarial loss weight.
        ar_gamma (float): AR-VAE weight.
        device (torch.device): Target device.
        total_step (int): Global step counter.
        wandb_run (Any): Active W&B run or None.
        ar_vae_enabled (bool): Whether AR-VAE is enabled.
        regularized_attributes (dict[str, Any] | None): AR-VAE config.
        pairwise_mode (str): Pairwise mode for AR-VAE.
        subset_pairs (int | None): Subset size for AR-VAE pairs.
        adv_enabled (bool): Whether adversarial training is enabled.
        max_epochs (int): Total epochs.

    Returns:
        int: Updated global step.
    """
    autoencoder.train()
    if discriminator is not None:
        discriminator.train()

    attribute_latent_mapping, delta_global = _resolve_ar_config(regularized_attributes)

    adv_active = _adv_is_active(
        adv_enabled=adv_enabled,
        adv_loss=adv_loss,
        discriminator=discriminator,
        epoch=epoch,
    )

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")):
        images, batch_attributes = _prepare_batch(batch, device, ar_vae_enabled)

        optimizer_g.zero_grad(set_to_none=True)
        outputs = _compute_generator_outputs(
            images=images,
            autoencoder=autoencoder,
            intensity_loss=intensity_loss,
            loss_perceptual=loss_perceptual,
            discriminator=discriminator,
            adv_loss=adv_loss,
            adv_enabled=adv_enabled,
            epoch=epoch,
            ar_vae_enabled=ar_vae_enabled,
            batch_attributes=batch_attributes,
            attribute_latent_mapping=attribute_latent_mapping,
            pairwise_mode=pairwise_mode,
            subset_pairs=subset_pairs,
            delta_global=delta_global,
            cast_recons_to_float=False,
            adv_active=adv_active,
        )

        loss_g = compute_total_loss(
            recons_loss=outputs.recons_loss,
            kl_loss=outputs.kl_loss,
            perceptual_loss=outputs.perceptual_loss,
            adv_gen_loss=outputs.adv_gen_loss,
            ar_loss=outputs.ar_loss,
            kl_weight=kl_weight,
            perceptual_weight=perceptual_weight,
            adv_weight=adv_weight,
            ar_gamma=ar_gamma,
            ar_vae_enabled=ar_vae_enabled,
        )

        loss_g.backward()
        optimizer_g.step()

        discriminator_loss = torch.tensor(0.0, device=device)
        if adv_active and optimizer_d is not None:
            optimizer_d.zero_grad(set_to_none=True)
            discriminator_loss = _compute_adv_disc_loss(
                outputs.reconstruction,
                images,
                discriminator,
                adv_loss,
                adv_enabled=adv_enabled,
                epoch=epoch,
            )
            loss_d = adv_weight * discriminator_loss
            loss_d.backward()
            optimizer_d.step()

        if wandb_run is not None:
            total_step += 1
            log_payload = {
                "train/recon_loss": outputs.recons_loss.item(),
                "train/kl_loss": outputs.kl_loss.item(),
                "train/perceptual_loss": outputs.perceptual_loss.item(),
                "train/adv_gen_loss": (adv_weight * outputs.adv_gen_loss).item() if adv_active else 0.0,
                "train/adv_disc_loss": (adv_weight * discriminator_loss).item() if adv_active else 0.0,
                "train/step": total_step,
                "train/loss_total": loss_g.item(),
            }
            if ar_vae_enabled:
                log_payload["train/ar_loss_total"] = outputs.ar_loss.item()
                for attr_name, loss_attr in outputs.ar_losses_per_attr.items():
                    log_payload[f"train/ar_loss_{attr_name}"] = loss_attr.item()

            wandb_run.log(log_payload, step=total_step)

            if step == 0:
                _log_triplet(
                    wandb_run,
                    tag="train/triplets",
                    image=images[0].detach().cpu(),
                    reconstruction=outputs.reconstruction[0].detach().cpu(),
                    step=total_step,
                    caption=f"train_epoch_{epoch:03d}_step_0",
                )

    return total_step


def _validate_epoch(
    *,
    epoch: int,
    val_loader: torch.utils.data.DataLoader,
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    intensity_loss: torch.nn.Module,
    loss_perceptual: PerceptualLoss,
    adv_loss: PatchAdversarialLoss | None,
    kl_weight: float,
    perceptual_weight: float,
    adv_weight: float,
    ar_gamma: float,
    device: torch.device,
    wandb_run: Any,
    log_triplet_every: int,
    ar_vae_enabled: bool,
    regularized_attributes: dict[str, Any] | None,
    pairwise_mode: str,
    subset_pairs: int | None,
    adv_enabled: bool,
) -> float:
    """Run one validation epoch.

    Args:
        epoch (int): Current epoch index.
        val_loader (torch.utils.data.DataLoader): Validation loader.
        autoencoder (torch.nn.Module): VAE model.
        discriminator (torch.nn.Module | None): Optional discriminator.
        intensity_loss (torch.nn.Module): Intensity loss.
        loss_perceptual (PerceptualLoss): Perceptual loss.
        adv_loss (PatchAdversarialLoss | None): Adversarial loss.
        kl_weight (float): KL weight.
        perceptual_weight (float): Perceptual weight.
        adv_weight (float): Adversarial weight.
        ar_gamma (float): AR-VAE weight.
        device (torch.device): Target device.
        wandb_run (Any): Active W&B run or None.
        log_triplet_every (int): Log triplets every N epochs.
        ar_vae_enabled (bool): Whether AR-VAE is enabled.
        regularized_attributes (dict[str, Any] | None): AR-VAE config.
        pairwise_mode (str): Pairwise mode for AR-VAE.
        subset_pairs (int | None): Subset size for AR-VAE pairs.
        adv_enabled (bool): Whether adversarial training is enabled.

    Returns:
        float: Validation reconstruction loss.
    """
    autoencoder.eval()
    if discriminator is not None:
        discriminator.eval()

    val_recon_epoch_loss = 0.0
    val_kl_epoch_loss = 0.0
    val_perc_epoch_loss = 0.0
    val_adv_gen_epoch_loss = 0.0
    val_adv_disc_epoch_loss = 0.0
    val_ar_epoch_loss = 0.0
    val_ar_losses_per_attr: dict[str, float] = {}

    attribute_latent_mapping, delta_global = _resolve_ar_config(regularized_attributes)

    triplet_logged = False
    adv_active = _adv_is_active(
        adv_enabled=adv_enabled,
        adv_loss=adv_loss,
        discriminator=discriminator,
        epoch=epoch,
    )

    for step, batch in enumerate(val_loader):
        images, batch_attributes = _prepare_batch(batch, device, ar_vae_enabled)

        with torch.no_grad():
            outputs = _compute_generator_outputs(
                images=images,
                autoencoder=autoencoder,
                intensity_loss=intensity_loss,
                loss_perceptual=loss_perceptual,
                discriminator=discriminator,
                adv_loss=adv_loss,
                adv_enabled=adv_enabled,
                epoch=epoch,
                ar_vae_enabled=ar_vae_enabled,
                batch_attributes=batch_attributes,
                attribute_latent_mapping=attribute_latent_mapping,
                pairwise_mode=pairwise_mode,
                subset_pairs=subset_pairs,
                delta_global=delta_global,
                cast_recons_to_float=True,
                adv_active=adv_active,
            )
            adv_disc_loss = torch.tensor(0.0, device=device)
            if adv_active:
                adv_disc_loss = _compute_adv_disc_loss(
                    outputs.reconstruction,
                    images,
                    discriminator,
                    adv_loss,
                    adv_enabled=adv_enabled,
                    epoch=epoch,
                )

        val_recon_epoch_loss += float(outputs.recons_loss.item())
        val_kl_epoch_loss += float(outputs.kl_loss.item())
        val_perc_epoch_loss += float(outputs.perceptual_loss.item())
        val_adv_gen_epoch_loss += float(outputs.adv_gen_loss.item())
        val_adv_disc_epoch_loss += float(adv_disc_loss.item())
        val_ar_epoch_loss += float(outputs.ar_loss.item())

        for attr_name, loss_attr in outputs.ar_losses_per_attr.items():
            val_ar_losses_per_attr[attr_name] = val_ar_losses_per_attr.get(attr_name, 0.0) + float(loss_attr.item())

        if wandb_run is not None and not triplet_logged and epoch % log_triplet_every == 0:
            _log_triplet(
                wandb_run,
                tag="val/triplets",
                image=images[0].detach().cpu(),
                reconstruction=outputs.reconstruction[0].detach().cpu(),
                caption=f"val_epoch_{epoch:03d}_step_{step:03d}",
            )
            triplet_logged = True

    denom = max(1, step + 1)
    val_recon_epoch_loss /= denom
    val_kl_epoch_loss /= denom
    val_perc_epoch_loss /= denom
    val_adv_gen_epoch_loss /= denom
    val_adv_disc_epoch_loss /= denom
    val_ar_epoch_loss /= denom
    val_ar_losses_per_attr = {k: v / denom for k, v in val_ar_losses_per_attr.items()}

    val_loss_total = compute_total_loss(
        recons_loss=torch.tensor(val_recon_epoch_loss),
        kl_loss=torch.tensor(val_kl_epoch_loss),
        perceptual_loss=torch.tensor(val_perc_epoch_loss),
        adv_gen_loss=torch.tensor(val_adv_gen_epoch_loss),
        ar_loss=torch.tensor(val_ar_epoch_loss),
        kl_weight=kl_weight,
        perceptual_weight=perceptual_weight,
        adv_weight=adv_weight,
        ar_gamma=ar_gamma,
        ar_vae_enabled=ar_vae_enabled,
    )

    if wandb_run is not None:
        log_dict = {
            "val/recon_loss": val_recon_epoch_loss,
            "val/kl_loss": val_kl_epoch_loss,
            "val/perceptual_loss": val_perc_epoch_loss,
            "val/adv_gen_loss": adv_weight * val_adv_gen_epoch_loss if adv_active else 0.0,
            "val/adv_disc_loss": adv_weight * val_adv_disc_epoch_loss if adv_active else 0.0,
            "val/loss_total": float(val_loss_total.item()),
            "epoch": epoch,
        }
        if ar_vae_enabled:
            log_dict["val/ar_loss_total"] = val_ar_epoch_loss
            for attr_name, loss_attr in val_ar_losses_per_attr.items():
                log_dict[f"val/ar_loss_{attr_name}"] = loss_attr
        wandb_run.log(log_dict)

    return val_recon_epoch_loss


def _save_last_checkpoint(
    *,
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    weights_dir: Path,
) -> None:
    """Save the latest generator (and discriminator if enabled).

    Args:
        autoencoder (torch.nn.Module): VAE model.
        discriminator (torch.nn.Module | None): Optional discriminator.
        weights_dir (Path): Directory to save weights.
    """
    torch.save(autoencoder.state_dict(), weights_dir / "autoencoder_last.pt")
    if discriminator is not None:
        torch.save(discriminator.state_dict(), weights_dir / "discriminator_last.pt")


def _save_best_checkpoint(
    *,
    epoch: int,
    autoencoder: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer | None,
    val_loss: float,
    total_step: int,
    best_val_loss: float,
    best_epoch_saved: int | None,
    weights_dir: Path,
) -> tuple[float, int | None]:
    """Save the best checkpoint if validation improves.

    Args:
        epoch (int): Current epoch.
        autoencoder (torch.nn.Module): VAE model.
        discriminator (torch.nn.Module | None): Optional discriminator.
        optimizer_g (torch.optim.Optimizer): Generator optimizer.
        optimizer_d (torch.optim.Optimizer | None): Discriminator optimizer.
        val_loss (float): Current validation loss.
        total_step (int): Global step counter.
        best_val_loss (float): Best validation loss so far.
        best_epoch_saved (int | None): Previously saved best epoch.
        weights_dir (Path): Directory for saving weights.

    Returns:
        tuple[float, int | None]: Updated best_val_loss and best_epoch_saved.
    """
    if val_loss >= best_val_loss:
        return best_val_loss, best_epoch_saved

    if best_epoch_saved is not None:
        for filename in [
            weights_dir / f"checkpoint_epoch{best_epoch_saved}.pth",
            weights_dir / f"autoencoder_epoch{best_epoch_saved}.pth",
            weights_dir / f"discriminator_epoch{best_epoch_saved}.pth",
        ]:
            if filename.exists():
                filename.unlink()

    torch.save(autoencoder.state_dict(), weights_dir / f"autoencoder_epoch{epoch}.pth")
    if discriminator is not None:
        torch.save(discriminator.state_dict(), weights_dir / f"discriminator_epoch{epoch}.pth")

    checkpoint_path = weights_dir / f"checkpoint_epoch{epoch}.pth"
    torch.save(
        {
            "epoch": epoch,
            "autoencoder_state_dict": autoencoder.state_dict(),
            "discriminator_state_dict": discriminator.state_dict() if discriminator is not None else None,
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict() if optimizer_d is not None else None,
            "best_val_loss": val_loss,
            "total_step": total_step,
        },
        checkpoint_path,
    )

    print(f"✅ Best models saved for epoch {epoch}")
    return val_loss, epoch


def train() -> None:
    """Entry point to train the VAE."""
    args = parse_args()
    config = load_config(args.config_file)
    apply_overrides(config, batch_size=args.batch_size, lr=args.lr, max_epochs=args.max_epochs)

    device = init_device_and_seed(args.seed, print_monai_config=False)

    run_dir = Path(config.run_dir)
    if run_dir.exists() and not getattr(config, "resume_ckpt", False):
        raise ValueError(
            f"Run directory already exists: {run_dir}\n"
            "To prevent overwriting previous runs:\n"
            "  1. Change 'run_dir' in your config file, or\n"
            "  2. Set 'resume_ckpt: true' to continue training"
        )

    run_dir, weights_dir, splits_dir = resolve_run_dirs(config.run_dir)

    regularized_attributes = getattr(config, "regularized_attributes", {})
    ar_from_train = _coerce_bool(config.autoencoder_train.get("ar_vae_enabled", False))
    ar_from_block = _coerce_bool(regularized_attributes.get("enabled", False))
    ar_vae_enabled = ar_from_train or ar_from_block

    pairwise_mode = regularized_attributes.get("pairwise", "all")
    subset_pairs = regularized_attributes.get("subset_pairs")
    ar_gamma = float(config.autoencoder_train.get("ar_vae_weight", regularized_attributes.get("gamma", 0.0)))

    train_loader, val_loader, train_paths, val_paths = create_train_val_dataloaders(
        data_base_dir=config.data_base_dir,
        batch_size=config.autoencoder_train["batch_size"],
        patch_size=tuple(config.autoencoder_train["patch_size"]),
        data_source=config.data_source,
        train_split=config.train_split,
        subset_size=config.autoencoder_train.get("subset_size"),
        seed=args.seed,
        val_dir=config.val_dir,
        ar_vae_enabled=ar_vae_enabled,
        regularized_attributes=regularized_attributes,
    )

    _save_split(
        splits_dir,
        seed=args.seed,
        train_split=config.train_split,
        val_dir=config.val_dir,
        train_paths=train_paths,
        val_paths=val_paths,
    )

    adv_enabled = _coerce_bool(config.autoencoder_train.get("adv_enabled", False))

    autoencoder, discriminator = _build_models(config, device, adv_enabled)
    if autoencoder is None:
        raise RuntimeError("Autoencoder initialization failed.")

    intensity_loss, adv_loss, loss_perceptual = _build_losses(
        config.autoencoder_train, device, config.spatial_dims, adv_enabled
    )
    optimizer_g, optimizer_d = _build_optimizers(
        autoencoder, discriminator, float(config.autoencoder_train["lr"]), adv_enabled
    )

    start_epoch, best_val_loss, total_step, best_epoch_saved = _load_checkpoint(
        config,
        autoencoder,
        discriminator,
        optimizer_g,
        optimizer_d,
        device,
    )

    wandb_cfg = getattr(config, "wandb", {"enabled": False})
    wandb_run = init_wandb(
        wandb_cfg,
        run_dir=run_dir,
        config_file=args.config_file,
        train_cfg=config.autoencoder_train,
        config=config,
    )

    kl_weight = float(config.autoencoder_train["kl_weight"])
    perceptual_weight = float(config.autoencoder_train["perceptual_weight"])
    adv_weight = float(config.autoencoder_train.get("adv_weight", 0.0))
    max_epochs = int(config.autoencoder_train["max_epochs"])
    val_interval = int(config.autoencoder_train["val_interval"])
    log_triplet_every = 20

    for epoch in range(start_epoch, max_epochs):
        start_time = time.time()

        total_step = _train_epoch(
            epoch=epoch,
            train_loader=train_loader,
            autoencoder=autoencoder,
            discriminator=discriminator,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            intensity_loss=intensity_loss,
            adv_loss=adv_loss,
            loss_perceptual=loss_perceptual,
            kl_weight=kl_weight,
            perceptual_weight=perceptual_weight,
            adv_weight=adv_weight,
            ar_gamma=ar_gamma,
            device=device,
            total_step=total_step,
            wandb_run=wandb_run,
            ar_vae_enabled=ar_vae_enabled,
            regularized_attributes=regularized_attributes,
            pairwise_mode=pairwise_mode,
            subset_pairs=subset_pairs,
            adv_enabled=adv_enabled,
            max_epochs=max_epochs,
        )

        if epoch % val_interval == 0:
            val_loss = _validate_epoch(
                epoch=epoch,
                val_loader=val_loader,
                autoencoder=autoencoder,
                discriminator=discriminator,
                intensity_loss=intensity_loss,
                loss_perceptual=loss_perceptual,
                adv_loss=adv_loss,
                kl_weight=kl_weight,
                perceptual_weight=perceptual_weight,
                adv_weight=adv_weight,
                ar_gamma=ar_gamma,
                device=device,
                wandb_run=wandb_run,
                log_triplet_every=log_triplet_every,
                ar_vae_enabled=ar_vae_enabled,
                regularized_attributes=regularized_attributes,
                pairwise_mode=pairwise_mode,
                subset_pairs=subset_pairs,
                adv_enabled=adv_enabled,
            )

            print(f"Epoch {epoch} val_loss: {val_loss:.4f} | Time: {time.time() - start_time:.1f}s")
            if wandb_run is not None:
                wandb_run.log({"time_per_epoch": time.time() - start_time})

            _save_last_checkpoint(autoencoder=autoencoder, discriminator=discriminator, weights_dir=weights_dir)
            best_val_loss, best_epoch_saved = _save_best_checkpoint(
                epoch=epoch,
                autoencoder=autoencoder,
                discriminator=discriminator,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                val_loss=val_loss,
                total_step=total_step,
                best_val_loss=best_val_loss,
                best_epoch_saved=best_epoch_saved,
                weights_dir=weights_dir,
            )

    if wandb_run is not None:
        try:
            import wandb  # type: ignore
        except ImportError:
            wandb = None
        if wandb is not None:
            wandb.finish()


def main() -> None:
    """CLI entry point for VAE training."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()


if __name__ == "__main__":
    main()
