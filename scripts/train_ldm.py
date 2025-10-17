#!/usr/bin/env python
"""Latent Diffusion Model (LDM) Training Script.

This script trains a conditional latent diffusion model for medical image synthesis. The model takes dental images as
conditioning input and generates edentulous images.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from monai.config import print_config
from monai.inferers import LatentDiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from monai.utils import first, set_determinism
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from pti_ldm_vae.data import create_ldm_dataloaders
from pti_ldm_vae.models import DiffusionUNet, VAEModel, create_condition_projector
from pti_ldm_vae.utils.distributed import setup_ddp
from pti_ldm_vae.utils.visualization import normalize_batch_for_display


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LDM Training Script")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./config/environment_tif.json",
        help="Environment json file that stores environment paths",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./config/config_train_16g_cond.json",
        help="Config json file that stores hyper-parameters",
    )
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

    return ddp_bool, rank, world_size, device, dist


def load_config(args):
    """Load and merge configuration files."""
    env_dict = json.load(open(args.environment_file))["ldm"]
    config_dict = json.load(open(args.config_file))

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)

    args.model_dir = os.path.join(args.run_dir, "trained_weights")
    args.tfevent_path = os.path.join(args.run_dir, "tfevent")

    return args


def load_vae_and_compute_scale(args, device, ddp_bool, dist, rank, train_loader):
    """Load pretrained VAE and compute latent space scale factor."""
    autoencoder = VAEModel.from_config(args.autoencoder_def).to(device)

    map_location = {"cuda:0": f"cuda:{rank}"}
    autoencoder.load_state_dict(torch.load(args.autoencoder_path, map_location=map_location, weights_only=True))
    print(f"Rank {rank}: Loaded pretrained autoencoder from {args.autoencoder_path}")

    # Compute scale factor
    with torch.no_grad(), autocast("cuda", enabled=True):
        check_data = first(train_loader)[0]
        z = autoencoder.encode_stage_2_inputs(check_data.to(device))

    scale_factor = 1 / torch.std(z)
    print(f"Rank {rank}: Local scale_factor: {scale_factor}")

    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)

    print(f"Rank {rank}: Final scale_factor: {scale_factor}")

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)

    return autoencoder, scale_factor, z


def create_diffusion_models(args, device, ddp_bool, rank):
    """Create UNet and condition projector."""
    unet = DiffusionUNet.from_config(args.diffusion_def).to(device)
    condition_projector = create_condition_projector(
        condition_input_dim=4, cross_attention_dim=args.diffusion_def["cross_attention_dim"]
    ).to(device)

    # Load checkpoint if requested
    if args.resume_ckpt:
        trained_diffusion_path = os.path.join(args.model_dir, "diffusion_unet.pt")
        map_location = {"cuda:0": f"cuda:{rank}"}
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location, weights_only=True))
            print(f"Rank {rank}: Loaded trained diffusion model from {trained_diffusion_path}")
        except FileNotFoundError:
            print(f"Rank {rank}: Train diffusion model from scratch")

    if ddp_bool:
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    return unet, condition_projector


def create_scheduler_and_inferer(args, scale_factor):
    """Create scheduler and inferer."""
    scheduler = DDPMScheduler(
        num_train_timesteps=args.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=args.NoiseScheduler["beta_start"],
        beta_end=args.NoiseScheduler["beta_end"],
    )
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    return scheduler, inferer


def train_epoch(
    epoch,
    train_loader,
    autoencoder,
    unet,
    condition_projector,
    inferer,
    optimizer_diff,
    scaler,
    z_shape,
    device,
    rank,
    total_step,
    tensorboard_writer,
    ddp_bool,
    max_epochs,
):
    """Train for one epoch."""
    unet.train()

    if ddp_bool:
        train_loader.sampler.set_epoch(epoch)
        inferer_autoencoder = autoencoder.module
    else:
        inferer_autoencoder = autoencoder

    for step, (images, condition_images) in enumerate(train_loader):
        if step % 100 == 0 and rank == 0:
            print(f"🌀 Epoch {epoch} - Batch {step}/{len(train_loader)}")

        images = images.to(device)
        condition_images = condition_images.to(device)

        # Encode condition images to latent space
        condition_latent = autoencoder.encode_stage_2_inputs(condition_images)
        B, C, H, W = condition_latent.shape
        condition_seq = condition_latent.permute(0, 2, 3, 1).reshape(B, H * W, C)
        condition_context = condition_projector(condition_seq)

        optimizer_diff.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=True):
            noise_shape = [images.shape[0], *list(z_shape[1:])]
            noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            noise_pred = inferer(
                inputs=images,
                autoencoder_model=inferer_autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps,
                condition=condition_context,
                mode="crossattn",
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        if rank == 0:
            total_step += 1
            tensorboard_writer.add_scalar("train_diffusion_loss_iter", loss, total_step)

            # Visualization at first step
            if step == 0:
                visualize_training(
                    images,
                    condition_images,
                    condition_context,
                    inferer,
                    inferer_autoencoder,
                    unet,
                    z_shape,
                    device,
                    tensorboard_writer,
                    total_step,
                )

    return total_step


def visualize_training(
    images,
    condition_images,
    condition_context,
    inferer,
    inferer_autoencoder,
    unet,
    z_shape,
    device,
    tensorboard_writer,
    total_step,
) -> None:
    """Visualize training progress."""
    with torch.no_grad():
        torch.manual_seed(42)
        np.random.seed(42)

        cond = condition_images[0].unsqueeze(0).detach().cpu()
        img = images[0].unsqueeze(0).detach().cpu()

        cond_disp = torch.rot90(normalize_batch_for_display(cond), k=3, dims=[2, 3])[0]
        img_disp = torch.rot90(normalize_batch_for_display(img), k=3, dims=[2, 3])[0]

        noise_shape = [1, *list(z_shape[1:])]
        noise = torch.randn(noise_shape, dtype=images.dtype).to(device)
        z = inferer_autoencoder.encode_stage_2_inputs(images[0:1].to(device))

        t_values = [100, 500, 999]
        grid_rows = [torch.cat([cond_disp, img_disp], dim=2)]

        for t_int in t_values:
            t = torch.tensor([t_int], device=device).long()
            z_t = inferer.scheduler.add_noise(original_samples=z, noise=noise, timesteps=t)

            alphas_cumprod = inferer.scheduler.alphas_cumprod.to(device)
            alpha_t = alphas_cumprod[t].reshape(-1, 1, 1, 1).type_as(z)

            noise_img = inferer_autoencoder.decode_stage_2_outputs(z_t).cpu()
            noise_img_disp = torch.rot90(normalize_batch_for_display(noise_img), k=3, dims=[2, 3])[0]

            noise_pred = unet(z_t, timesteps=t, context=condition_context[0:1])
            z_hat = (z_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            recon = inferer_autoencoder.decode_stage_2_outputs(z_hat).cpu()
            recon_disp = torch.rot90(normalize_batch_for_display(recon), k=3, dims=[2, 3])[0]

            row = torch.cat([noise_img_disp, recon_disp], dim=2)
            grid_rows.append(row)

        full_visu = torch.cat(grid_rows, dim=1)
        tensorboard_writer.add_image("train_visu_4x2_noise_recon", full_visu, global_step=total_step)


def validate(
    epoch,
    val_loader,
    autoencoder,
    unet,
    condition_projector,
    inferer,
    scheduler,
    z_shape,
    args,
    device,
    rank,
    ddp_bool,
    tensorboard_writer,
):
    """Run validation."""
    autoencoder.eval()
    unet.eval()
    val_recon_epoch_loss = 0
    quatuors = []
    val_steps_used = 0

    start_epoch_to_save = 10
    save_every = 2
    save_root = Path(os.path.join(args.run_dir, "validation_samples"))
    do_save_images = rank == 0 and epoch >= start_epoch_to_save and epoch % save_every == 0

    if do_save_images:
        epoch_dir = save_root / f"epoch_{epoch}"
        dir_image = epoch_dir / "edente"
        dir_synth = epoch_dir / "edente_synth"
        dir_image.mkdir(parents=True, exist_ok=True)
        dir_synth.mkdir(parents=True, exist_ok=True)

    inferer_autoencoder = autoencoder.module if ddp_bool else autoencoder

    with torch.no_grad(), autocast("cuda", enabled=True):
        for step, (images, condition_images) in enumerate(val_loader):
            if step % 7 != 0:
                continue

            if rank == 0:
                print(f"🔬 Validation epoch {epoch} - step {step}/{len(val_loader)}")

            images = images.to(device)
            condition_images = condition_images.to(device)

            condition_latent = autoencoder.encode_stage_2_inputs(condition_images)
            B, C, H, W = condition_latent.shape
            condition_seq = condition_latent.permute(0, 2, 3, 1).reshape(B, H * W, C)
            condition_context = condition_projector(condition_seq)

            noise_shape = [images.shape[0], *list(z_shape[1:])]
            noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            noise_pred = inferer(
                inputs=images,
                autoencoder_model=inferer_autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps,
                condition=condition_context[0:1, ...],
                mode="crossattn",
            )
            val_loss = F.mse_loss(noise_pred.float(), noise.float())
            val_recon_epoch_loss += val_loss
            val_steps_used += 1

            if rank == 0:
                synthetic_images = inferer.sample(
                    input_noise=noise[0:1, ...],
                    autoencoder_model=inferer_autoencoder,
                    diffusion_model=unet,
                    scheduler=scheduler,
                    conditioning=condition_context[0:1, ...],
                    mode="crossattn",
                )

                cond = condition_images[0].unsqueeze(0).detach().cpu()
                img = images[0].unsqueeze(0).detach().cpu()
                recon = synthetic_images[0].unsqueeze(0).detach().cpu()
                noise_cpu = noise[0].unsqueeze(0).detach().cpu()

                noise_img = torch.clamp(
                    inferer_autoencoder.decode_stage_2_outputs(noise_cpu.to(device)), min=-5, max=5
                ).cpu()

                if do_save_images:
                    tifffile.imwrite(
                        dir_image / f"step{step:03}.tif", torch.rot90(img[0].squeeze(), k=3, dims=[0, 1]).numpy()
                    )
                    tifffile.imwrite(
                        dir_synth / f"step{step:03}.tif", torch.rot90(recon[0].squeeze(), k=3, dims=[0, 1]).numpy()
                    )

                cond_disp = torch.rot90(normalize_batch_for_display(cond), k=3, dims=[2, 3])[0]
                img_disp = torch.rot90(normalize_batch_for_display(img), k=3, dims=[2, 3])[0]
                noise_disp = torch.rot90(normalize_batch_for_display(noise_img), k=3, dims=[2, 3])[0]
                recon_disp = torch.rot90(normalize_batch_for_display(recon), k=3, dims=[2, 3])[0]
                quatuor = torch.cat([cond_disp, img_disp, noise_disp, recon_disp], dim=2)
                quatuors.append((step, quatuor))

    val_recon_epoch_loss = val_recon_epoch_loss / val_steps_used

    if rank == 0:
        tensorboard_writer.add_scalar("val_diffusion_loss", val_recon_epoch_loss, epoch)
        for step_idx, quatuor in quatuors:
            tensorboard_writer.add_image(f"val_quatuor/step{step_idx:03}", quatuor, global_step=epoch)

    return val_recon_epoch_loss


def save_checkpoint(
    epoch,
    unet,
    condition_projector,
    optimizer_diff,
    val_loss,
    total_step,
    best_val_loss,
    best_epoch_saved,
    args,
    ddp_bool,
    rank,
):
    """Save model checkpoints."""
    if rank != 0 or epoch < 10:
        return best_val_loss, best_epoch_saved

    # Save last
    trained_diffusion_path_last = os.path.join(args.model_dir, "diffusion_unet_last.pt")
    if ddp_bool:
        torch.save(unet.module.state_dict(), trained_diffusion_path_last)
    else:
        torch.save(unet.state_dict(), trained_diffusion_path_last)

    # Save best
    if val_loss < best_val_loss:
        if best_epoch_saved is not None:
            files_to_remove = [
                os.path.join(args.model_dir, f"checkpoint_epoch{best_epoch_saved}.pth"),
                os.path.join(args.model_dir, f"diffusion_unet_epoch{best_epoch_saved}.pth"),
            ]
            for f in files_to_remove:
                if os.path.exists(f):
                    os.remove(f)

        torch.save(
            unet.module.state_dict() if ddp_bool else unet.state_dict(),
            os.path.join(args.model_dir, f"diffusion_unet_epoch{epoch}.pth"),
        )

        torch.save(
            {
                "epoch": epoch,
                "unet_state_dict": unet.module.state_dict() if ddp_bool else unet.state_dict(),
                "condition_projector_state_dict": condition_projector.state_dict(),
                "optimizer_state_dict": optimizer_diff.state_dict(),
                "best_val_loss": val_loss,
                "total_step": total_step,
            },
            os.path.join(args.model_dir, f"checkpoint_epoch{epoch}.pth"),
        )

        print(f"✅ Best models saved for epoch {epoch}")
        return val_loss, epoch

    return best_val_loss, best_epoch_saved


def main() -> None:
    args = parse_args()
    ddp_bool, rank, world_size, device, dist = setup_environment(args)
    args = load_config(args)

    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)

    set_determinism(42)

    # Create dataloaders
    train_loader, val_loader = create_ldm_dataloaders(
        data_base_dir=args.data_base_dir,
        batch_size=args.diffusion_train["batch_size"],
        patch_size=tuple(args.diffusion_train["patch_size"]),
        augment=args.augment,
        rank=rank,
    )

    # Load VAE and compute scale factor
    autoencoder, scale_factor, z = load_vae_and_compute_scale(args, device, ddp_bool, dist, rank, train_loader)

    # Create diffusion models
    unet, condition_projector = create_diffusion_models(args, device, ddp_bool, rank)

    # Create scheduler and inferer
    scheduler, inferer = create_scheduler_and_inferer(args, scale_factor)

    # Create optimizer
    optimizer_diff = torch.optim.Adam(
        list(unet.parameters()) + list(condition_projector.parameters()), lr=args.diffusion_train["lr"] * world_size
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_diff,
        milestones=args.diffusion_train["lr_scheduler_milestones"],
        gamma=0.1,
    )

    # Tensorboard
    if rank == 0:
        tensorboard_writer = SummaryWriter(args.tfevent_path)

    # Training setup
    autoencoder.eval()
    scaler = GradScaler("cuda")
    total_step = 0
    best_val_loss = 100.0
    best_epoch_saved = None
    max_epochs = args.diffusion_train["max_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    start_epoch = args.start_epoch if args.resume_ckpt else 0

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        start_time = time.time()
        if rank == 0:
            print(f"\n📅 Starting epoch {epoch}/{max_epochs - 1}")

        lr_scheduler.step()

        total_step = train_epoch(
            epoch,
            train_loader,
            autoencoder,
            unet,
            condition_projector,
            inferer,
            optimizer_diff,
            scaler,
            z.shape,
            device,
            rank,
            total_step,
            tensorboard_writer if rank == 0 else None,
            ddp_bool,
            max_epochs,
        )

        # Validation
        if epoch % val_interval == 0:
            if ddp_bool:
                val_loader.sampler.set_epoch(epoch)

            if rank == 0:
                print(f"\n🔍 Starting validation at epoch {epoch}")

            val_loss = validate(
                epoch,
                val_loader,
                autoencoder,
                unet,
                condition_projector,
                inferer,
                scheduler,
                z.shape,
                args,
                device,
                rank,
                ddp_bool,
                tensorboard_writer if rank == 0 else None,
            )

            if rank == 0:
                print(f"Epoch {epoch} val_loss: {val_loss:.4f} | Time: {time.time() - start_time:.1f}s")

            best_val_loss, best_epoch_saved = save_checkpoint(
                epoch,
                unet,
                condition_projector,
                optimizer_diff,
                val_loss,
                total_step,
                best_val_loss,
                best_epoch_saved,
                args,
                ddp_bool,
                rank,
            )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
