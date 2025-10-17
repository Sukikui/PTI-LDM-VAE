import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from monai.config import print_config
from monai.inferers import LatentDiffusionInferer
from monai.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism
from PIL import Image
from torch.amp import autocast
from tqdm import tqdm

from pti_ldm_vae.data import create_ldm_dataloaders
from pti_ldm_vae.models import DiffusionUNet, VAEModel, create_condition_projector
from pti_ldm_vae.utils.visualization import normalize_batch_for_display


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LDM Inference Script")
    parser.add_argument(
        "-e", "--environment-file", default="./config/environment_tif.json", help="Environment json file"
    )
    parser.add_argument("-c", "--config-file", default="./config/config_train_16g_cond.json", help="Config json file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., checkpoint_epoch50.pth)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: inference_<checkpoint_name>)"
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to generate (default: 10)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Any:
    """Load configuration from JSON files."""
    env_dict = json.load(open(args.environment_file))["ldm"]
    config_dict = json.load(open(args.config_file))

    class Config:
        pass

    config = Config()
    for k, v in env_dict.items():
        setattr(config, k, v)
    for k, v in config_dict.items():
        setattr(config, k, v)

    return config


def setup_output_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Create output directories."""
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(f"inference_{checkpoint_path.stem}") if args.output_dir is None else Path(args.output_dir)

    out_tif = output_dir / "results_tif"
    out_png = output_dir / "results_png"
    out_tif.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    return output_dir, out_tif, out_png


def load_models(config: Any, checkpoint_path: str, device: torch.device) -> tuple[VAEModel, DiffusionUNet, Any]:
    """Load VAE, UNet, and condition projector."""
    # Load VAE
    autoencoder = VAEModel.from_config(config.autoencoder_def).to(device)
    autoencoder.load_state_dict(torch.load(config.autoencoder_path, map_location=device, weights_only=True))
    autoencoder.eval()

    # Load UNet and condition projector
    unet = DiffusionUNet.from_config(config.diffusion_def).to(device)
    condition_projector = create_condition_projector(
        condition_input_dim=4, cross_attention_dim=config.diffusion_def["cross_attention_dim"]
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(checkpoint["unet_state_dict"])
    condition_projector.load_state_dict(checkpoint["condition_projector_state_dict"])
    unet.eval()

    return autoencoder, unet, condition_projector


def compute_scale_factor(
    autoencoder: VAEModel, val_loader: Any, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute latent space scale factor."""
    with torch.no_grad(), autocast("cuda", enabled=True):
        first_batch = next(iter(val_loader))
        check_data = first_batch[0]
        z = autoencoder.encode_stage_2_inputs(check_data.to(device))

    scale_factor = 1 / torch.std(z)
    return scale_factor, z


def create_scheduler_and_inferer(config: Any, scale_factor: float) -> tuple[DDPMScheduler, LatentDiffusionInferer]:
    """Create scheduler and inferer."""
    scheduler = DDPMScheduler(
        num_train_timesteps=config.NoiseScheduler["num_train_timesteps"],
        schedule="scaled_linear_beta",
        beta_start=config.NoiseScheduler["beta_start"],
        beta_end=config.NoiseScheduler["beta_end"],
    )
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    return scheduler, inferer


def generate_and_save_sample(
    idx: int, cond: torch.Tensor, target: torch.Tensor, synth: torch.Tensor, out_tif: Path, out_png: Path
) -> None:
    """Save a single generated sample as TIF and PNG."""
    # Extract numpy arrays
    cond_np = cond[0, 0].numpy()
    target_np = target[0, 0].numpy()
    synth_np = synth[0, 0].numpy()

    # Rotate for correct orientation
    cond_np = np.rot90(cond_np, k=3)
    target_np = np.rot90(target_np, k=3)
    synth_np = np.rot90(synth_np, k=3)

    # Save TIF: [condition | target | synthetic]
    concat_tif = np.concatenate([cond_np, target_np, synth_np], axis=1)
    tifffile.imwrite(out_tif / f"sample{idx:03d}.tif", concat_tif)

    # Save PNG (normalized for visualization)
    cond_disp = torch.rot90(normalize_batch_for_display(cond), k=3, dims=[2, 3])[0]
    target_disp = torch.rot90(normalize_batch_for_display(target), k=3, dims=[2, 3])[0]
    synth_disp = torch.rot90(normalize_batch_for_display(synth), k=3, dims=[2, 3])[0]

    # Concatenate [condition | target | synthetic]
    concat_png = torch.cat([cond_disp, target_disp, synth_disp], dim=2)
    array = (concat_png.numpy()[0] * 255).astype(np.uint8)
    Image.fromarray(array).save(out_png / f"sample{idx:03d}.png")


def run_inference(
    autoencoder: VAEModel,
    unet: DiffusionUNet,
    condition_projector: Any,
    inferer: LatentDiffusionInferer,
    scheduler: DDPMScheduler,
    val_loader: Any,
    z_shape: tuple,
    device: torch.device,
    num_samples: int,
    out_tif: Path,
    out_png: Path,
) -> None:
    """Run inference and generate samples."""
    num_generated = 0

    for _batch_idx, (target_images, condition_images) in enumerate(tqdm(val_loader, desc="Generating")):
        if num_generated >= num_samples:
            break

        target_images = target_images.to(device)
        condition_images = condition_images.to(device)

        with torch.no_grad(), autocast("cuda", enabled=True):
            # Encode condition images to latent space
            condition_latent = autoencoder.encode_stage_2_inputs(condition_images)
            B, C, H, W = condition_latent.shape
            condition_seq = condition_latent.permute(0, 2, 3, 1).reshape(B, H * W, C)
            condition_context = condition_projector(condition_seq)

            # Generate noise and sample
            noise_shape = [B, *list(z_shape[1:])]
            noise = torch.randn(noise_shape, dtype=target_images.dtype).to(device)

            synthetic_images = inferer.sample(
                input_noise=noise,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                scheduler=scheduler,
                conditioning=condition_context,
                mode="crossattn",
            )

        # Save results for each image in batch
        for i in range(B):
            if num_generated >= num_samples:
                break

            cond = condition_images[i].unsqueeze(0).detach().cpu()
            target = target_images[i].unsqueeze(0).detach().cpu()
            synth = synthetic_images[i].unsqueeze(0).detach().cpu()

            generate_and_save_sample(num_generated, cond, target, synth, out_tif, out_png)
            num_generated += 1


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print_config()
    set_determinism(42)

    # Load configuration
    config = load_config(args)

    # Setup output directories
    output_dir, out_tif, out_png = setup_output_dirs(args)
    print(f"[INFO] Output directory: {output_dir}")

    # Load models
    print("[INFO] Loading models...")
    autoencoder, unet, condition_projector = load_models(config, args.checkpoint, device)
    print(f"[INFO] Loaded VAE from {config.autoencoder_path}")
    print(f"[INFO] Loaded checkpoint from {args.checkpoint}")

    # Create dataloader
    _, val_loader = create_ldm_dataloaders(
        data_base_dir=config.data_base_dir,
        batch_size=args.batch_size,
        patch_size=tuple(config.diffusion_train["patch_size"]),
        augment=False,
        rank=0,
    )

    # Compute scale factor
    print("[INFO] Computing scale factor...")
    scale_factor, z = compute_scale_factor(autoencoder, val_loader, device)
    print(f"[INFO] Scale factor: {scale_factor}")

    # Create scheduler and inferer
    scheduler, inferer = create_scheduler_and_inferer(config, scale_factor)

    # Run inference
    print(f"[INFO] Generating {args.num_samples} samples...")
    run_inference(
        autoencoder,
        unet,
        condition_projector,
        inferer,
        scheduler,
        val_loader,
        z.shape,
        device,
        args.num_samples,
        out_tif,
        out_png,
    )

    print(f"✅ Inference complete. Results saved in: {output_dir}")
    print(f"   - TIF files: {out_tif}")
    print(f"   - PNG files: {out_png}")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
