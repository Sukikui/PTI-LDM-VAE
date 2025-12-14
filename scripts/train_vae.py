import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import tifffile
import torch
from dotenv import load_dotenv
from monai.bundle import ConfigParser
from monai.config import print_config
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from pti_ldm_vae.data import create_vae_dataloaders
from pti_ldm_vae.models import VAEModel, compute_ar_vae_loss, compute_kl_loss, compute_total_loss
from pti_ldm_vae.utils import ensure_three_channels
from pti_ldm_vae.utils.distributed import setup_ddp
from pti_ldm_vae.utils.visualization import normalize_batch_for_display

# Load environment variables from .env file
load_dotenv()


def parse_args():
    """Parse command line arguments (simplified)."""
    parser = argparse.ArgumentParser(description="VAE Training Script - Simplified Configuration")

    # Required: unified config file
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/ar_vae_edente.json",
        help="Path to unified JSON configuration file",
    )

    # Essential overrides
    parser.add_argument("-g", "--gpus", default=1, type=int, help="Number of GPUs for training (default: 1)")
    parser.add_argument("--batch-size", type=int, help="Override batch size from config")
    parser.add_argument("--lr", type=float, help="Override learning rate from config")
    parser.add_argument("--max-epochs", type=int, help="Override max epochs from config")

    # Performance options (kept from CLI for convenience)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--cache-rate",
        type=float,
        default=0.0,
        help="Fraction of data to cache in RAM, 0.0-1.0 (default: 0.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    # Debug option
    parser.add_argument("--subset-size", type=int, help="Use only N images for debugging")

    return parser.parse_args()


def setup_environment(args):
    """Setup distributed training environment and device."""
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()

    ddp_bool = args.gpus > 1
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if use_cuda else "cpu")
        dist = None

    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    print_config()
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    return ddp_bool, rank, world_size, device, dist


def load_config(args):
    """Load unified configuration file with CLI overrides using MONAI ConfigParser."""
    # Use MONAI's ConfigParser to handle @ references automatically
    parser = ConfigParser()
    parser.read_config(args.config_file)
    parser.parse(True)  # Parse @ references

    config = parser.get_parsed_content()

    # Apply all config values to args
    for k, v in config.items():
        setattr(args, k, v)

    # Apply CLI overrides (only essential ones)
    if args.batch_size:
        args.autoencoder_train["batch_size"] = args.batch_size
    if args.max_epochs:
        args.autoencoder_train["max_epochs"] = args.max_epochs
    if args.lr:
        args.autoencoder_train["lr"] = args.lr

    # Set derived paths
    args.model_dir = os.path.join(args.run_dir, "trained_weights")

    return args


def init_wandb(args, rank):
    """Initialize Weights & Biases logging."""
    if rank != 0 or not args.wandb.get("enabled", True):
        return None

    # Get W&B config from args and env
    project = os.getenv("WANDB_PROJECT", args.wandb.get("project", "pti-ldm-vae"))
    entity = args.wandb.get("entity") or os.getenv("WANDB_ENTITY")

    # Auto-generate run name from run_dir if not provided
    run_name = args.wandb.get("name")
    if run_name is None:
        run_name = Path(args.run_dir).name

    tags = args.wandb.get("tags", [])
    notes = args.wandb.get("notes", "")

    # Initialize W&B
    wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        notes=notes,
        config={
            "architecture": "VAE",
            "spatial_dims": args.spatial_dims,
            "latent_channels": args.latent_channels,
            "channels": args.autoencoder_def["channels"],
            "num_res_blocks": args.autoencoder_def["num_res_blocks"],
            "batch_size": args.autoencoder_train["batch_size"],
            "lr": args.autoencoder_train["lr"],
            "max_epochs": args.autoencoder_train["max_epochs"],
            "kl_weight": args.autoencoder_train["kl_weight"],
            "perceptual_weight": args.autoencoder_train["perceptual_weight"],
            "recon_loss": args.autoencoder_train["recon_loss"],
            "adv_weight": args.autoencoder_train["adv_weight"],
            "data_source": args.data_source,
            "augment": args.augment,
        },
    )

    # Log full config for traceability
    try:
        with open(args.config_file, encoding="utf-8") as cfg_file:
            full_cfg = cfg_file.read()
        wandb.config.update({"full_config_json": full_cfg}, allow_val_change=True)
        artifact = wandb.Artifact("vae-config", type="config")
        artifact.add_file(args.config_file)
        wandb.log_artifact(artifact)
    except Exception as exc:
        print(f"[WARN] Could not upload config file to W&B: {exc}")

    print(f"✨ W&B run initialized: {wandb.run.url}")
    return wandb


def _prepare_batch(
    batch: torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]] | list[torch.Tensor],
    device: torch.device,
    ar_vae_enabled: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
    """Convert a dataloader batch to tensors on the target device.

    This guards against environments where the default collate returns lists instead of stacked tensors.

    Args:
        batch: Batch from the dataloader (tensor, list of tensors, or tuple with attributes).
        device: Target device for training.
        ar_vae_enabled: Whether attribute-regularized VAE is active (requires attributes).

    Returns:
        Tuple of (images tensor, optional attributes dictionary on device).

    Raises:
        ValueError: If AR-VAE is enabled but attributes are missing.
        TypeError: If the batch format is unsupported.
    """
    images: torch.Tensor
    batch_attributes: dict[str, torch.Tensor] | None = None

    # Normalize batch structure; handle edge case where a list of (image, attrs) slips through
    if isinstance(batch, list):
        if not batch:
            raise ValueError("Empty batch received from dataloader.")
        if all(isinstance(item, tuple) and len(item) == 2 for item in batch):
            imgs: list[torch.Tensor] = []
            attr_buf: dict[str, list[torch.Tensor]] = {}
            for img, attrs in batch:
                imgs.append(torch.as_tensor(img))
                if attrs is not None:
                    for k, v in attrs.items():
                        attr_buf.setdefault(k, []).append(torch.as_tensor(v))
            images = torch.stack(imgs, dim=0)
            if attr_buf:
                batch_attributes = {k: torch.stack(v_list, dim=0).to(torch.float32) for k, v_list in attr_buf.items()}
        elif len(batch) == 2 and isinstance(batch[0], torch.Tensor) and isinstance(batch[1], dict):
            images = batch[0]
            batch_attributes = {k: torch.as_tensor(v) for k, v in batch[1].items()}
        else:
            raise TypeError(f"Unsupported list batch elements: {[type(item) for item in batch]}")
    else:
        if isinstance(batch, tuple):
            images, batch_attributes = batch
        else:
            images = batch

    if not isinstance(images, torch.Tensor):
        raise TypeError(f"Unsupported batch type: {type(images)}")

    images = images.to(device)

    if batch_attributes is not None:
        batch_attributes = {k: v.to(device) for k, v in batch_attributes.items()}
    elif ar_vae_enabled:
        raise ValueError("AR-VAE is enabled but attributes are missing from the batch.")

    return images, batch_attributes


def _resolve_bool(value: Any) -> bool:
    """Interpret strings like \"false\"/\"true\" safely instead of Python truthiness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
        return False  # Unrecognized strings default to False to avoid accidental activation
    if value is None:
        return False
    return bool(value)


def create_models(args, device, ddp_bool, rank):
    """Create VAE and discriminator models."""
    autoencoder = VAEModel.from_config(args.autoencoder_def).to(device)

    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)

    if ddp_bool:
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[device], output_device=rank, find_unused_parameters=True)

    return autoencoder, discriminator


def create_losses_and_optimizers(args, autoencoder, discriminator, device, world_size, rank):
    """Create loss functions and optimizers."""
    if args.autoencoder_train.get("recon_loss") == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Using L2 loss")
    else:
        intensity_loss = L1Loss()
        if rank == 0:
            print("Using L1 loss")

    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=args.spatial_dims, network_type="squeeze").to(device)

    optimizer_g = torch.optim.Adam(autoencoder.parameters(), lr=args.autoencoder_train["lr"] * world_size)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.autoencoder_train["lr"] * world_size)

    return intensity_loss, adv_loss, loss_perceptual, optimizer_g, optimizer_d


def load_checkpoint(args, autoencoder, discriminator, optimizer_g, optimizer_d, device, ddp_bool):
    """Load checkpoint if resume is requested."""
    if args.resume_ckpt:
        checkpoint_path = args.checkpoint_dir
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(f"cuda:{device}"))

            if ddp_bool:
                autoencoder.module.load_state_dict(checkpoint["autoencoder_state_dict"])
                discriminator.module.load_state_dict(checkpoint["discriminator_state_dict"])
            else:
                autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
                discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

            optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
            optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])

            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            total_step = checkpoint["total_step"]

            print(f"[INFO] Resuming from epoch {start_epoch} | best_val_loss = {best_val_loss:.4f}")
            return start_epoch, best_val_loss, total_step, checkpoint["epoch"]
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {checkpoint_path}")
    print("[INFO] Training from scratch")
    return 0, 100.0, 0, None


def train_epoch(
    epoch,
    train_loader,
    autoencoder,
    discriminator,
    optimizer_g,
    optimizer_d,
    intensity_loss,
    adv_loss,
    loss_perceptual,
    kl_weight,
    perceptual_weight,
    adv_weight,
    device,
    rank,
    total_step,
    use_wandb,
    ddp_bool,
    max_epochs,
    ar_vae_enabled,
    regularized_attributes,
    pairwise_mode,
    subset_pairs,
    ar_gamma,
):
    """Train for one epoch."""
    autoencoder.train()
    discriminator.train()

    if ddp_bool:
        train_loader.sampler.set_epoch(epoch)

    raw_mapping = regularized_attributes.get("attribute_latent_mapping", {}) if regularized_attributes else {}
    attribute_latent_mapping = {k: v for k, v in raw_mapping.items() if not str(k).startswith("_")}
    delta_global = regularized_attributes.get("delta_global", {}) if regularized_attributes else {}

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")):
        images, batch_attributes = _prepare_batch(batch, device, ar_vae_enabled)

        # Train generator
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_logvar = autoencoder(images)

        latent_vectors = z_mu
        if latent_vectors.dim() == 4:
            latent_vectors = latent_vectors.mean(dim=(2, 3))
        elif latent_vectors.dim() != 2:
            raise ValueError(f"Unexpected latent shape: {latent_vectors.shape}")

        recons_loss = intensity_loss(reconstruction, images)
        kl_loss = compute_kl_loss(z_mu, z_logvar)
        recon_rgb = ensure_three_channels(reconstruction.float())
        images_rgb = ensure_three_channels(images.float())
        p_loss = loss_perceptual(recon_rgb, images_rgb)
        generator_loss = torch.tensor(0.0, device=device)

        if epoch > 5:  # warmup epochs
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

        ar_loss = torch.tensor(0.0, device=device)
        ar_losses_per_attr: dict[str, torch.Tensor] = {}
        ar_pairs: dict[str, int] = {}
        ar_deltas: dict[str, float] = {}
        if ar_vae_enabled:
            ar_loss, ar_losses_per_attr, ar_pairs, ar_deltas = compute_ar_vae_loss(
                latent_vectors=latent_vectors,
                attributes=batch_attributes if batch_attributes is not None else {},
                attribute_latent_mapping=attribute_latent_mapping,
                pairwise_mode=pairwise_mode,
                subset_pairs=subset_pairs,
                delta_global=delta_global,
            )
        else:
            ar_loss = torch.tensor(0.0, device=device)

        loss_g = compute_total_loss(
            recons_loss=recons_loss,
            kl_loss=kl_loss,
            perceptual_loss=p_loss,
            adv_gen_loss=generator_loss,
            ar_loss=ar_loss,
            kl_weight=kl_weight,
            perceptual_weight=perceptual_weight,
            adv_weight=adv_weight,
            ar_gamma=ar_gamma,
            ar_vae_enabled=ar_vae_enabled,
        )

        # Debug stats for KL explosion (first batch only)
        if step == 0 and rank == 0:
            with torch.no_grad():
                sigma = torch.exp(0.5 * z_logvar)
                print(
                    "[DEBUG] Train batch0 stats | "
                    f"z_mu mean={z_mu.mean().item():.4f} min={z_mu.min().item():.4f} max={z_mu.max().item():.4f} | "
                    f"logvar mean={z_logvar.mean().item():.4f} min={z_logvar.min().item():.4f} max={z_logvar.max().item():.4f} | "
                    f"sigma mean={sigma.mean().item():.4f} min={sigma.min().item():.4f} max={sigma.max().item():.4f} | "
                    f"kl_loss={kl_loss.item():.4f}"
                )

        loss_g.backward()
        optimizer_g.step()

        # Train discriminator
        discriminator_loss = torch.tensor(0.0, device=device)
        if epoch > 5:
            optimizer_d.zero_grad(set_to_none=True)
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = adv_weight * discriminator_loss
            loss_d.backward()
            optimizer_d.step()

        # Logging
        if rank == 0 and use_wandb:
            total_step += 1
            log_payload = {
                "train/recon_loss": intensity_loss(reconstruction, images).item(),
                "train/kl_loss": kl_loss.item(),
                "train/perceptual_loss": p_loss.item(),
                "train/adv_gen_loss": (adv_weight * generator_loss).item() if epoch > 5 else 0.0,
                "train/adv_disc_loss": (adv_weight * discriminator_loss).item() if epoch > 5 else 0.0,
                "train/step": total_step,
                "train/loss_total": loss_g.item(),
            }
            if ar_vae_enabled:
                log_payload["train/ar_loss_total"] = ar_loss.item()
                for attr_name, loss_attr in ar_losses_per_attr.items():
                    log_payload[f"train/ar_loss_{attr_name}"] = loss_attr.item()

            wandb.log(log_payload, step=total_step)

            if step == 0:
                with torch.no_grad():
                    img = images[0].unsqueeze(0).detach().cpu()
                    recon = reconstruction[0].unsqueeze(0).detach().cpu()
                    diff = torch.abs(img - recon)

                    img_disp = torch.rot90(normalize_batch_for_display(img), k=3, dims=[2, 3])[0]
                    recon_disp = torch.rot90(normalize_batch_for_display(recon), k=3, dims=[2, 3])[0]
                    diff_disp = torch.rot90(normalize_batch_for_display(diff), k=3, dims=[2, 3])[0]

                    triplet = torch.cat([img_disp, recon_disp, diff_disp], dim=2)
                    wandb.log(
                        {"train/triplets": [wandb.Image(triplet.permute(1, 2, 0).numpy(), caption="train_step_0")]},
                        step=total_step,
                    )

    return total_step


def validate(
    epoch,
    val_loader,
    autoencoder,
    discriminator,
    intensity_loss,
    loss_perceptual,
    perceptual_weight,
    adv_loss,
    adv_weight,
    kl_weight,
    ar_gamma,
    args,
    device,
    rank,
    ddp_bool,
    use_wandb,
    log_triplet_every,
    ar_vae_enabled,
    regularized_attributes,
    pairwise_mode,
    subset_pairs,
):
    """Run validation."""
    autoencoder.eval()
    if discriminator is not None:
        discriminator.eval()
    val_recon_epoch_loss = 0
    val_kl_epoch_loss = 0
    val_perc_epoch_loss = 0
    val_adv_gen_epoch_loss = 0
    val_adv_disc_epoch_loss = 0
    val_ar_epoch_loss = 0
    val_ar_losses_per_attr: dict[str, float] = {}
    triplets = []
    max_triplets_to_log = 1

    # Saving parameters
    start_epoch_to_save = 10
    save_every = 5
    save_root = Path(os.path.join(args.run_dir, "validation_samples"))
    do_save_images = rank == 0 and epoch >= start_epoch_to_save and epoch % save_every == 0

    if do_save_images:
        epoch_dir = save_root / f"epoch_{epoch}"
        dir_original = epoch_dir / "originale"
        dir_recon = epoch_dir / "reconstruction"
        dir_diff = epoch_dir / "diff"
        dir_original.mkdir(parents=True, exist_ok=True)
        dir_recon.mkdir(parents=True, exist_ok=True)
        dir_diff.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(val_loader):
        images, batch_attributes = _prepare_batch(batch, device, ar_vae_enabled)

        with torch.no_grad():
            reconstruction, z_mu, z_logvar = autoencoder(images)
            recon_rgb = ensure_three_channels(reconstruction.float())
            images_rgb = ensure_three_channels(images.float())
            p_loss = loss_perceptual(recon_rgb, images_rgb)
            recons_loss = intensity_loss(reconstruction.float(), images.float())
            kl_loss = compute_kl_loss(z_mu, z_logvar)

            adv_gen_loss = torch.tensor(0.0, device=device)
            adv_disc_loss = torch.tensor(0.0, device=device)
            if discriminator is not None and epoch > 5:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                adv_gen_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                logits_fake_detached = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake_detached, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                adv_disc_loss = (loss_d_fake + loss_d_real) * 0.5

            ar_loss = torch.tensor(0.0, device=device)
            ar_losses_per_attr: dict[str, torch.Tensor] = {}
            if ar_vae_enabled:
                raw_mapping = (
                    regularized_attributes.get("attribute_latent_mapping", {}) if regularized_attributes else {}
                )
                attribute_latent_mapping = {k: v for k, v in raw_mapping.items() if not str(k).startswith("_")}
                delta_global = regularized_attributes.get("delta_global", {}) if regularized_attributes else {}
                ar_loss, ar_losses_per_attr, _, _ = compute_ar_vae_loss(
                    latent_vectors=z_mu,
                    attributes=batch_attributes if batch_attributes is not None else {},
                    attribute_latent_mapping=attribute_latent_mapping,
                    pairwise_mode=pairwise_mode,
                    subset_pairs=subset_pairs,
                    delta_global=delta_global,
                )

        # Debug stats for KL explosion (first val batch only)
        if step == 0 and rank == 0:
            sigma = torch.exp(0.5 * z_logvar)
            print(
                "[DEBUG] Val batch0 stats | "
                f"z_mu mean={z_mu.mean().item():.4f} min={z_mu.min().item():.4f} max={z_mu.max().item():.4f} | "
                f"logvar mean={z_logvar.mean().item():.4f} min={z_logvar.min().item():.4f} max={z_logvar.max().item():.4f} | "
                f"sigma mean={sigma.mean().item():.4f} min={sigma.min().item():.4f} max={sigma.max().item():.4f} | "
                f"kl_loss={kl_loss.item():.4f}"
            )

        val_recon_epoch_loss += recons_loss.item()
        val_kl_epoch_loss += kl_loss.item()
        val_perc_epoch_loss += p_loss.item()
        val_adv_gen_epoch_loss += adv_gen_loss.item()
        val_adv_disc_epoch_loss += (adv_weight * adv_disc_loss).item()
        val_ar_epoch_loss += ar_loss.item()
        for attr_name, loss_attr in ar_losses_per_attr.items():
            val_ar_losses_per_attr[attr_name] = val_ar_losses_per_attr.get(attr_name, 0.0) + loss_attr.item()

        if rank == 0:
            img = images[0].squeeze().detach().cpu()
            recon = reconstruction[0].squeeze().detach().cpu()
            diff = torch.abs(img - recon)

            if do_save_images:
                tifffile.imwrite(dir_original / f"step{step:03}.tif", torch.rot90(img, k=3, dims=[0, 1]).numpy())
                tifffile.imwrite(dir_recon / f"step{step:03}.tif", torch.rot90(recon, k=3, dims=[0, 1]).numpy())
                tifffile.imwrite(dir_diff / f"step{step:03}.tif", torch.rot90(diff, k=3, dims=[0, 1]).numpy())

            img_disp = torch.rot90(normalize_batch_for_display(img.unsqueeze(0).unsqueeze(0)), k=3, dims=[2, 3])[0]
            recon_disp = torch.rot90(normalize_batch_for_display(recon.unsqueeze(0).unsqueeze(0)), k=3, dims=[2, 3])[0]
            diff_disp = torch.rot90(normalize_batch_for_display(diff.unsqueeze(0).unsqueeze(0)), k=3, dims=[2, 3])[0]

            triplet = torch.cat([img_disp, recon_disp, diff_disp], dim=2)
            if len(triplets) < max_triplets_to_log and epoch % log_triplet_every == 0:
                triplets.append((step, triplet))

    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
    val_kl_epoch_loss = val_kl_epoch_loss / (step + 1)
    val_perc_epoch_loss = val_perc_epoch_loss / (step + 1)
    val_adv_gen_epoch_loss = val_adv_gen_epoch_loss / (step + 1)
    val_adv_disc_epoch_loss = val_adv_disc_epoch_loss / (step + 1)
    val_ar_epoch_loss = val_ar_epoch_loss / (step + 1)
    val_ar_losses_per_attr = {k: v / (step + 1) for k, v in val_ar_losses_per_attr.items()}

    val_loss_total = compute_total_loss(
        recons_loss=val_recon_epoch_loss,
        kl_loss=val_kl_epoch_loss,
        perceptual_loss=val_perc_epoch_loss,
        adv_gen_loss=val_adv_gen_epoch_loss,
        ar_loss=val_ar_epoch_loss,
        kl_weight=kl_weight,
        perceptual_weight=perceptual_weight,
        adv_weight=adv_weight,
        ar_gamma=ar_gamma,
        ar_vae_enabled=ar_vae_enabled,
    )

    if rank == 0 and use_wandb:
        log_dict = {
            "val/recon_loss": val_recon_epoch_loss,  # intensity only
            "val/kl_loss": val_kl_epoch_loss,
            "val/perceptual_loss": val_perc_epoch_loss,
            "val/adv_gen_loss": adv_weight * val_adv_gen_epoch_loss,
            "val/adv_disc_loss": val_adv_disc_epoch_loss,
            "val/loss_total": val_loss_total,
            "epoch": epoch,
        }
        if ar_vae_enabled:
            log_dict["val/ar_loss_total"] = val_ar_epoch_loss
            for attr_name, loss_attr in val_ar_losses_per_attr.items():
                log_dict[f"val/ar_loss_{attr_name}"] = loss_attr
        if epoch % log_triplet_every == 0:
            images = [
                wandb.Image(triplet.permute(1, 2, 0).numpy(), caption=f"step{step_idx:03}")
                for step_idx, triplet in triplets
            ]
            if images:
                log_dict["val/triplets"] = images
        wandb.log(log_dict)

    return val_recon_epoch_loss


def save_checkpoint(
    epoch,
    autoencoder,
    discriminator,
    optimizer_g,
    optimizer_d,
    val_loss,
    total_step,
    best_epoch_saved,
    args,
    ddp_bool,
    rank,
):
    """Save model checkpoints."""
    if rank != 0:
        return best_epoch_saved

    # Save last
    trained_g_path_last = os.path.join(args.model_dir, "autoencoder_last.pt")
    trained_d_path_last = os.path.join(args.model_dir, "discriminator_last.pt")

    if ddp_bool:
        torch.save(autoencoder.module.state_dict(), trained_g_path_last)
        torch.save(discriminator.module.state_dict(), trained_d_path_last)
    else:
        torch.save(autoencoder.state_dict(), trained_g_path_last)
        torch.save(discriminator.state_dict(), trained_d_path_last)

    return best_epoch_saved


def save_best_checkpoint(
    epoch,
    autoencoder,
    discriminator,
    optimizer_g,
    optimizer_d,
    val_loss,
    total_step,
    best_val_loss,
    best_epoch_saved,
    args,
    ddp_bool,
    rank,
):
    """Save best model checkpoint."""
    if rank != 0:
        return best_val_loss, best_epoch_saved

    if val_loss >= best_val_loss:
        return best_val_loss, best_epoch_saved

    # Clean old best
    if best_epoch_saved is not None:
        files_to_remove = [
            os.path.join(args.model_dir, f"checkpoint_epoch{best_epoch_saved}.pth"),
            os.path.join(args.model_dir, f"autoencoder_epoch{best_epoch_saved}.pth"),
            os.path.join(args.model_dir, f"discriminator_epoch{best_epoch_saved}.pth"),
        ]
        for f in files_to_remove:
            if os.path.exists(f):
                os.remove(f)

    # Save new best
    if ddp_bool:
        torch.save(autoencoder.module.state_dict(), os.path.join(args.model_dir, f"autoencoder_epoch{epoch}.pth"))
        torch.save(discriminator.module.state_dict(), os.path.join(args.model_dir, f"discriminator_epoch{epoch}.pth"))
    else:
        torch.save(autoencoder.state_dict(), os.path.join(args.model_dir, f"autoencoder_epoch{epoch}.pth"))
        torch.save(discriminator.state_dict(), os.path.join(args.model_dir, f"discriminator_epoch{epoch}.pth"))

    checkpoint_path = os.path.join(args.model_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save(
        {
            "epoch": epoch,
            "autoencoder_state_dict": autoencoder.module.state_dict() if ddp_bool else autoencoder.state_dict(),
            "discriminator_state_dict": discriminator.module.state_dict() if ddp_bool else discriminator.state_dict(),
            "optimizer_g_state_dict": optimizer_g.state_dict(),
            "optimizer_d_state_dict": optimizer_d.state_dict(),
            "best_val_loss": val_loss,
            "total_step": total_step,
        },
        checkpoint_path,
    )

    print(f"✅ Best models saved for epoch {epoch}")

    return val_loss, epoch


def main() -> None:
    args = parse_args()
    ddp_bool, rank, world_size, device, dist = setup_environment(args)
    args = load_config(args)
    regularized_attributes = (
        getattr(args, "regularized_attributes", {}) if hasattr(args, "regularized_attributes") else {}
    )
    ar_from_train = _resolve_bool(args.autoencoder_train.get("ar_vae_enabled", False))
    ar_from_block = _resolve_bool(regularized_attributes.get("enabled", False))
    ar_vae_enabled = ar_from_train or ar_from_block
    pairwise_mode = regularized_attributes.get("pairwise", "all")
    subset_pairs = regularized_attributes.get("subset_pairs")
    raw_gamma = args.autoencoder_train.get("ar_vae_weight", regularized_attributes.get("gamma", 0.0))
    if isinstance(raw_gamma, str):
        # Handle unresolved references like "@regularized_attributes.gamma" gracefully
        try:
            ar_gamma = float(raw_gamma)
        except ValueError:
            ar_gamma = float(regularized_attributes.get("gamma", 0.0))
    else:
        ar_gamma = float(raw_gamma)

    # Check if run_dir already exists
    if rank == 0:
        run_dir = Path(args.run_dir)
        if run_dir.exists() and not args.resume_ckpt:
            raise ValueError(
                f"Run directory already exists: {run_dir}\n"
                f"To prevent overwriting previous runs:\n"
                f"  1. Change 'run_dir' in your config file, or\n"
                f"  2. Set 'resume_ckpt: true' to continue training"
            )
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        splits_dir = run_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

    set_determinism(args.seed)

    # Create dataloaders with new parameters
    train_loader, val_loader, train_paths, val_paths = create_vae_dataloaders(
        data_base_dir=args.data_base_dir,
        batch_size=args.autoencoder_train["batch_size"],
        patch_size=tuple(args.autoencoder_train["patch_size"]),
        augment=args.augment,
        rank=rank,
        data_source=args.data_source,
        train_split=args.train_split,
        num_workers=args.num_workers,
        seed=args.seed,
        subset_size=args.subset_size,
        val_dir=args.val_dir,
        cache_rate=args.cache_rate,
        distributed=ddp_bool,
        world_size=world_size,
        ar_vae_enabled=ar_vae_enabled,
        regularized_attributes=regularized_attributes,
    )

    # Persist split files (rank 0 only)
    if rank == 0:
        split_payload = {
            "seed": args.seed,
            "train_split": args.train_split,
            "subset_size": args.subset_size,
            "val_dir": args.val_dir,
            "train_files": list(train_paths),
            "val_files": list(val_paths),
        }
        split_path = Path(args.run_dir) / "splits" / "vae_split.json"
        with open(split_path, "w", encoding="utf-8") as split_file:
            json.dump(split_payload, split_file, indent=2)
        print(f"[INFO] Saved train/val split to {split_path}")

    # Create models
    autoencoder, discriminator = create_models(args, device, ddp_bool, rank)

    if rank == 0:
        model_to_print = autoencoder.module if ddp_bool else autoencoder
        print("\n=== Autoencoder model summary ===")
        print(model_to_print)
        print("=================================\n")

    # Create losses and optimizers
    intensity_loss, adv_loss, loss_perceptual, optimizer_g, optimizer_d = create_losses_and_optimizers(
        args, autoencoder, discriminator, device, world_size, rank
    )

    # Load checkpoint
    start_epoch, best_val_loss, total_step, best_epoch_saved = load_checkpoint(
        args, autoencoder, discriminator, optimizer_g, optimizer_d, device, ddp_bool
    )

    # Initialize W&B
    use_wandb = init_wandb(args, rank) is not None

    # Define W&B metrics to allow different step semantics
    if use_wandb and rank == 0:
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("epoch")
        wandb.define_metric("time_per_epoch", step_metric="epoch")

    # Training parameters
    kl_weight = args.autoencoder_train["kl_weight"]
    perceptual_weight = args.autoencoder_train["perceptual_weight"]
    adv_weight = float(args.autoencoder_train["adv_weight"])
    max_epochs = args.autoencoder_train["max_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    log_triplet_every = 20

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        start_time = time.time()

        total_step = train_epoch(
            epoch,
            train_loader,
            autoencoder,
            discriminator,
            optimizer_g,
            optimizer_d,
            intensity_loss,
            adv_loss,
            loss_perceptual,
            kl_weight,
            perceptual_weight,
            adv_weight,
            device,
            rank,
            total_step,
            use_wandb,
            ddp_bool,
            max_epochs,
            ar_vae_enabled,
            regularized_attributes,
            pairwise_mode,
            subset_pairs,
            ar_gamma,
        )

        # Validation
        if epoch % val_interval == 0:
            if ddp_bool:
                val_loader.sampler.set_epoch(epoch)

            val_loss = validate(
                epoch,
                val_loader,
                autoencoder,
                discriminator,
                intensity_loss,
                loss_perceptual,
                perceptual_weight,
                adv_loss,
                adv_weight,
                kl_weight,
                ar_gamma,
                args,
                device,
                rank,
                ddp_bool,
                use_wandb,
                log_triplet_every,
                ar_vae_enabled,
                regularized_attributes,
                pairwise_mode,
                subset_pairs,
            )

            if rank == 0:
                print(f"Epoch {epoch} val_loss: {val_loss:.4f} | Time: {time.time() - start_time:.1f}s")
                if use_wandb:
                    wandb.log({"time_per_epoch": time.time() - start_time})

            best_epoch_saved = save_checkpoint(
                epoch,
                autoencoder,
                discriminator,
                optimizer_g,
                optimizer_d,
                val_loss,
                total_step,
                best_epoch_saved,
                args,
                ddp_bool,
                rank,
            )

            best_val_loss, best_epoch_saved = save_best_checkpoint(
                epoch,
                autoencoder,
                discriminator,
                optimizer_g,
                optimizer_d,
                val_loss,
                total_step,
                best_val_loss,
                best_epoch_saved,
                args,
                ddp_bool,
                rank,
            )

    # Finish W&B run
    if rank == 0 and use_wandb:
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
