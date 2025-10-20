import argparse
import logging
import os
import sys
import time
from pathlib import Path

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
from pti_ldm_vae.models import VAEModel, compute_kl_loss
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
        default="./config/vae_config.json",
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
            "data_source": args.data_source,
            "augment": args.augment,
        },
    )

    print(f"✨ W&B run initialized: {wandb.run.url}")
    return wandb


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
):
    """Train for one epoch."""
    autoencoder.train()
    discriminator.train()

    if ddp_bool:
        train_loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}")):
        images = batch.to(device)

        # Train generator
        optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = autoencoder(images)

        recons_loss = intensity_loss(reconstruction, images)
        kl_loss = compute_kl_loss(z_mu, z_sigma)
        p_loss = loss_perceptual(reconstruction.float(), images.float())
        loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

        if epoch > 5:  # warmup epochs
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = loss_g + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        # Train discriminator
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
            wandb.log({"train/recon_loss": recons_loss.item(), "train/step": total_step}, step=total_step)

            if step == 0:
                with torch.no_grad():
                    img = images[0].unsqueeze(0).detach().cpu()
                    recon = reconstruction[0].unsqueeze(0).detach().cpu()
                    diff = torch.abs(img - recon)

                    img_disp = torch.rot90(normalize_batch_for_display(img), k=3, dims=[2, 3])[0]
                    recon_disp = torch.rot90(normalize_batch_for_display(recon), k=3, dims=[2, 3])[0]
                    diff_disp = torch.rot90(normalize_batch_for_display(diff), k=3, dims=[2, 3])[0]

                    triplet = torch.cat([img_disp, recon_disp, diff_disp], dim=2)
                    wandb.log({"train/triplets": wandb.Image(triplet.permute(1, 2, 0).numpy())}, step=total_step)

    return total_step


def validate(
    epoch,
    val_loader,
    autoencoder,
    intensity_loss,
    loss_perceptual,
    perceptual_weight,
    args,
    device,
    rank,
    ddp_bool,
    use_wandb,
):
    """Run validation."""
    autoencoder.eval()
    val_recon_epoch_loss = 0
    triplets = []

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
        images = batch.to(device)

        with torch.no_grad():
            reconstruction, z_mu, z_sigma = autoencoder(images)
            recons_loss = intensity_loss(reconstruction.float(), images.float()) + perceptual_weight * loss_perceptual(
                reconstruction.float(), images.float()
            )

        val_recon_epoch_loss += recons_loss.item()

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
            triplets.append((step, triplet))

    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)

    if rank == 0 and use_wandb:
        log_dict = {"val/recon_loss": val_recon_epoch_loss, "epoch": epoch}
        for step_idx, triplet in triplets:
            log_dict[f"val/triplet_step{step_idx:03}"] = wandb.Image(triplet.permute(1, 2, 0).numpy())
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

    set_determinism(args.seed)

    # Create dataloaders with new parameters
    train_loader, val_loader = create_vae_dataloaders(
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
    )

    # Create models
    autoencoder, discriminator = create_models(args, device, ddp_bool, rank)

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
    adv_weight = 0.5
    max_epochs = args.autoencoder_train["max_epochs"]
    val_interval = args.autoencoder_train["val_interval"]

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
        )

        # Validation
        if epoch % val_interval == 0:
            if ddp_bool:
                val_loader.sampler.set_epoch(epoch)

            val_loss = validate(
                epoch,
                val_loader,
                autoencoder,
                intensity_loss,
                loss_perceptual,
                perceptual_weight,
                args,
                device,
                rank,
                ddp_bool,
                use_wandb,
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
