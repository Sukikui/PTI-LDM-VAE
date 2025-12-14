import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from monai.config import print_config
from monai.utils import set_determinism
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from pti_ldm_vae.data import create_vae_inference_dataloader
from pti_ldm_vae.models import VAEModel
from pti_ldm_vae.utils.vae_loader import load_vae_config, load_vae_model
from pti_ldm_vae.utils.visualization import normalize_batch_for_display


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VAE Inference Script")
    parser.add_argument("-c", "--config-file", required=True, help="Config json file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., checkpoint_epoch73.pth)"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input TIF images")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: inference_<checkpoint_name>)"
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    return parser.parse_args()


def load_config(config_file: str) -> Any:
    """Load configuration from JSON file."""
    return load_vae_config(config_file)


def setup_output_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Create output directories."""
    checkpoint_path = Path(args.checkpoint)
    if args.output_dir is None:
        checkpoint_name = checkpoint_path.stem
        output_dir = Path(f"inference_vae_{checkpoint_name}")
    else:
        output_dir = Path(args.output_dir)

    out_tif = output_dir / "results_tif"
    out_png = output_dir / "results_png"
    out_tif.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)

    return output_dir, out_tif, out_png


def create_dataloader(
    input_dir: str,
    patch_size: tuple,
    batch_size: int,
    num_samples: int | None = None,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, int]:
    """Create dataloader for inference.

    Args:
        input_dir (str): Directory containing .tif images.
        patch_size (tuple): Spatial resize target as (H, W).
        batch_size (int): Batch size for iteration.
        num_samples (int | None): Optional cap on number of images to process.
        num_workers (int): Number of worker processes for data loading.

    Returns:
        tuple[DataLoader, int]: Dataloader and number of images found.

    Raises:
        FileNotFoundError: If no .tif images are discovered in ``input_dir``.
    """
    dataloader, image_paths = create_vae_inference_dataloader(
        input_dir=input_dir,
        patch_size=patch_size,
        batch_size=batch_size,
        num_samples=num_samples,
        num_workers=num_workers,
    )
    return dataloader, len(image_paths)


def load_model(config: Any, checkpoint_path: str, device: torch.device) -> VAEModel:
    """Load VAE model from checkpoint."""
    return load_vae_model(config, checkpoint_path, device)


def reconstruct_deterministic(autoencoder: VAEModel, images: torch.Tensor) -> torch.Tensor:
    """Reconstruct images using deterministic latent mean (z_mu)."""
    z_mu = autoencoder.encode_deterministic(images)
    return autoencoder.decode_stage_2_outputs(z_mu)


def save_results(idx: int, input_img: torch.Tensor, recon_img: torch.Tensor, out_tif: Path, out_png: Path) -> None:
    """Save a single result as TIF and PNG."""
    # Get numpy arrays (remove batch and channel dimensions)
    input_np = input_img[0].numpy()
    recon_np = recon_img[0].numpy()

    # Save TIF: [original | reconstruction] side by side
    concat_tif = np.concatenate([input_np, recon_np], axis=1)
    tifffile.imwrite(out_tif / f"image{idx:04d}.tif", concat_tif)

    # Save PNG (normalized for visualization)
    input_disp = normalize_batch_for_display(input_img.unsqueeze(0))[0]
    recon_disp = normalize_batch_for_display(recon_img.unsqueeze(0))[0]
    concat_png = torch.cat([input_disp, recon_disp], dim=2)
    array = (concat_png.numpy()[0] * 255).astype(np.uint8)
    Image.fromarray(array).save(out_png / f"image{idx:04d}.png")


def run_inference(
    autoencoder: VAEModel, dataloader: DataLoader, device: torch.device, out_tif: Path, out_png: Path
) -> None:
    """Run inference and save results."""
    img_idx = 0

    for batch in tqdm(dataloader, desc="Processing"):
        with torch.no_grad():
            images = batch.to(device)
            reconstruction = reconstruct_deterministic(autoencoder, images)

            images = images.cpu()
            reconstruction = reconstruction.cpu()

        # Save each image in batch
        for i in range(images.shape[0]):
            save_results(img_idx, images[i], reconstruction[i], out_tif, out_png)
            img_idx += 1


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print_config()
    set_determinism(42)

    # Load configuration
    config = load_config(args.config_file)
    print(f"[INFO] Loaded config from {args.config_file}")

    # Setup output directories
    output_dir, out_tif, out_png = setup_output_dirs(args)
    print(f"[INFO] Output directory: {output_dir}")

    # Create dataloader
    patch_size = tuple(config.autoencoder_train["patch_size"])
    dataloader, num_images = create_dataloader(
        input_dir=args.input_dir,
        patch_size=patch_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Found {num_images} images in {args.input_dir}")

    # Load model
    print("[INFO] Loading VAE model...")
    autoencoder = load_model(config, args.checkpoint, device)
    print(f"[INFO] Loaded checkpoint from {args.checkpoint}")

    # Run inference
    print(f"[INFO] Running inference on {num_images} images...")
    run_inference(autoencoder, dataloader, device, out_tif, out_png)

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
