import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import tifffile
import torch
from monai.config import print_config
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism
from pti_ldm_vae.data import create_vae_dataloaders
from pti_ldm_vae.models import VAEModel, compute_kl_loss
from pti_ldm_vae.utils.distributed import setup_ddp
from pti_ldm_vae.utils.visualization import normalize_batch_for_display
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VAE Training Script")
    parser.add_argument("-e", "--environment-file", default="./config/environment_tif.json",
                        help="Environment json file that stores environment paths")
    parser.add_argument("-c", "--config-file", default="./config/config_train_16g_cond.json",
                        help="Config json file that stores hyper-parameters")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="Number of GPUs per node")
    return parser.parse_args()


def setup_environment(args):
    """Setup distributed training environment and device."""
    ddp_bool = args.gpus > 1
    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0
        dist = None

    torch.cuda.set_device(device)
    print(f"Using device: {device}")

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    return ddp_bool, rank, world_size, device, dist


def load_config(args):
    """Load and merge configuration files."""
    env_dict = json.load(open(args.environment_file, "r"))["vae"]
    config_dict = json.load(open(args.config_file, "r"))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    args.model_dir = os.path.join(args.run_dir, "trained_weights")
    args.tfevent_path = os.path.join(args.run_dir, "tfevent")

    return args


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
        else:
            raise FileNotFoundError(f"[ERROR] Checkpoint not found: {checkpoint_path}")
    else:
        print("[INFO] Training from scratch")
        return 0, 100.0, 0, None


def train_epoch(epoch, train_loader, autoencoder, discriminator, optimizer_g, optimizer_d,
                intensity_loss, adv_loss, loss_perceptual, kl_weight, perceptual_weight,
                adv_weight, device, rank, total_step, tensorboard_writer, ddp_bool, max_epochs):
    """Train for one epoch."""
    autoencoder.train()
    discriminator.train()

    if ddp_bool:
        train_loader.sampler.set_epoch(epoch)

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")):
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
        if rank == 0:
            total_step += 1
            tensorboard_writer.add_scalar("train_recon_loss_iter", recons_loss, total_step)

            if step == 0:
                with torch.no_grad():
                    img = images[0].unsqueeze(0).detach().cpu()
                    recon = reconstruction[0].unsqueeze(0).detach().cpu()
                    diff = torch.abs(img - recon)

                    img_disp = torch.rot90(normalize_batch_for_display(img), k=3, dims=[2, 3])[0]
                    recon_disp = torch.rot90(normalize_batch_for_display(recon), k=3, dims=[2, 3])[0]
                    diff_disp = torch.rot90(normalize_batch_for_display(diff), k=3, dims=[2, 3])[0]

                    triplet = torch.cat([img_disp, recon_disp, diff_disp], dim=2)
                    tensorboard_writer.add_image("train_triplets", triplet, global_step=total_step)

    return total_step


def validate(epoch, val_loader, autoencoder, intensity_loss, loss_perceptual, perceptual_weight,
             args, device, rank, ddp_bool, tensorboard_writer):
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
            recons_loss = intensity_loss(reconstruction.float(), images.float()) + \
                          perceptual_weight * loss_perceptual(reconstruction.float(), images.float())

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

    if rank == 0:
        tensorboard_writer.add_scalar("val_recon_loss", val_recon_epoch_loss, epoch)
        for step_idx, triplet in triplets:
            tensorboard_writer.add_image(f"val_triplets/step{step_idx:03}", triplet, global_step=epoch)

    return val_recon_epoch_loss


def save_checkpoint(epoch, autoencoder, discriminator, optimizer_g, optimizer_d,
                   val_loss, total_step, best_epoch_saved, args, ddp_bool, rank):
    """Save model checkpoints."""
    if rank != 0 or epoch < 10:
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


def save_best_checkpoint(epoch, autoencoder, discriminator, optimizer_g, optimizer_d,
                        val_loss, total_step, best_val_loss, best_epoch_saved, args, ddp_bool, rank):
    """Save best model checkpoint."""
    if rank != 0 or epoch < 10:
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
        torch.save(autoencoder.module.state_dict(),
                  os.path.join(args.model_dir, f"autoencoder_epoch{epoch}.pth"))
        torch.save(discriminator.module.state_dict(),
                  os.path.join(args.model_dir, f"discriminator_epoch{epoch}.pth"))
    else:
        torch.save(autoencoder.state_dict(),
                  os.path.join(args.model_dir, f"autoencoder_epoch{epoch}.pth"))
        torch.save(discriminator.state_dict(),
                  os.path.join(args.model_dir, f"discriminator_epoch{epoch}.pth"))

    checkpoint_path = os.path.join(args.model_dir, f"checkpoint_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'autoencoder_state_dict': autoencoder.module.state_dict() if ddp_bool else autoencoder.state_dict(),
        'discriminator_state_dict': discriminator.module.state_dict() if ddp_bool else discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'best_val_loss': val_loss,
        'total_step': total_step,
    }, checkpoint_path)

    print(f"✅ Best models saved for epoch {epoch}")

    return val_loss, epoch


def main():
    args = parse_args()
    ddp_bool, rank, world_size, device, dist = setup_environment(args)
    args = load_config(args)

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)

    set_determinism(42)

    # Create dataloaders
    train_loader, val_loader = create_vae_dataloaders(
        data_base_dir=args.data_base_dir,
        batch_size=args.autoencoder_train["batch_size"],
        patch_size=tuple(args.autoencoder_train["patch_size"]),
        augment=args.augment,
        rank=rank,
        data_source="edente",
    )

    # Create models
    autoencoder, discriminator = create_models(args, device, ddp_bool, rank)

    # Create losses and optimizers
    intensity_loss, adv_loss, loss_perceptual, optimizer_g, optimizer_d = \
        create_losses_and_optimizers(args, autoencoder, discriminator, device, world_size, rank)

    # Load checkpoint
    start_epoch, best_val_loss, total_step, best_epoch_saved = \
        load_checkpoint(args, autoencoder, discriminator, optimizer_g, optimizer_d, device, ddp_bool)

    # Tensorboard
    if rank == 0:
        tensorboard_writer = SummaryWriter(args.tfevent_path)

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
            epoch, train_loader, autoencoder, discriminator, optimizer_g, optimizer_d,
            intensity_loss, adv_loss, loss_perceptual, kl_weight, perceptual_weight,
            adv_weight, device, rank, total_step, tensorboard_writer if rank == 0 else None,
            ddp_bool, max_epochs
        )

        # Validation
        if epoch % val_interval == 0:
            if ddp_bool:
                val_loader.sampler.set_epoch(epoch)

            val_loss = validate(
                epoch, val_loader, autoencoder, intensity_loss, loss_perceptual,
                perceptual_weight, args, device, rank, ddp_bool,
                tensorboard_writer if rank == 0 else None
            )

            if rank == 0:
                print(f"Epoch {epoch} val_loss: {val_loss:.4f} | Time: {time.time() - start_time:.1f}s")

            best_epoch_saved = save_checkpoint(
                epoch, autoencoder, discriminator, optimizer_g, optimizer_d,
                val_loss, total_step, best_epoch_saved, args, ddp_bool, rank
            )

            best_val_loss, best_epoch_saved = save_best_checkpoint(
                epoch, autoencoder, discriminator, optimizer_g, optimizer_d,
                val_loss, total_step, best_val_loss, best_epoch_saved, args, ddp_bool, rank
            )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()