from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from tqdm import tqdm

from .config import load_config_and_model
from .data import create_inference_dataloader
from pti_ldm_vae_v2.vae_regression_common import VAEModel
from pti_ldm_vae_v2.vae_regression_common import init_device_and_seed, resolve_run_output_dir
from .visualization import normalize_batch_for_display


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VAE inference.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="VAE Inference Script")
    parser.add_argument("-c", "--config-file", required=True, help="Path to config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Directory with input TIF images.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def resolve_output_dirs(run_dir: str | Path, input_dir: str, output_dir: str | None) -> tuple[Path, Path, Path]:
    """Resolve output directories for VAE inference results.

    Args:
        run_dir (str | Path): VAE run directory from the config.
        input_dir (str): Input directory passed to the inference command.
        output_dir (str | None): Optional override for output directory.

    Returns:
        tuple[Path, Path, Path]: Root output directory, TIF subfolder, PNG subfolder.
    """
    base_output = resolve_run_output_dir(run_dir, input_dir, output_dir, "inference")

    out_tif = base_output / "results_tif"
    out_png = base_output / "results_png"
    out_tif.mkdir(parents=True, exist_ok=True)
    out_png.mkdir(parents=True, exist_ok=True)
    return base_output, out_tif, out_png


def save_results(idx: int, input_img: torch.Tensor, recon_img: torch.Tensor, out_tif: Path, out_png: Path) -> None:
    """Save a single result as TIF and PNG.

    Args:
        idx (int): Global image index for naming outputs.
        input_img (torch.Tensor): Original input image tensor without batch dimension.
        recon_img (torch.Tensor): Reconstructed image tensor without batch dimension.
        out_tif (Path): Destination directory for TIF outputs.
        out_png (Path): Destination directory for PNG outputs.
    """
    input_np = input_img[0].numpy()
    recon_np = recon_img[0].numpy()

    concat_tif = np.concatenate([input_np, recon_np], axis=1)
    tifffile.imwrite(out_tif / f"image{idx:04d}.tif", concat_tif)

    input_disp = normalize_batch_for_display(input_img.unsqueeze(0))[0]
    recon_disp = normalize_batch_for_display(recon_img.unsqueeze(0))[0]
    concat_png = torch.cat([input_disp, recon_disp], dim=2)
    array = (concat_png.numpy()[0] * 255).astype(np.uint8)
    Image.fromarray(array).save(out_png / f"image{idx:04d}.png")


def run_inference(
    autoencoder: VAEModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    out_tif: Path,
    out_png: Path,
) -> None:
    """Run inference and save results.

    Args:
        autoencoder (VAEModel): Loaded VAE model.
        dataloader (torch.utils.data.DataLoader): Dataloader yielding images.
        device (torch.device): Target device.
        out_tif (Path): Directory for TIF outputs.
        out_png (Path): Directory for PNG outputs.
    """
    img_idx = 0
    for batch in tqdm(dataloader, desc="Processing"):
        with torch.no_grad():
            images = batch.to(device)
            reconstruction = autoencoder.reconstruct_deterministic(images)

            images = images.cpu()
            reconstruction = reconstruction.cpu()

        for i in range(images.shape[0]):
            save_results(img_idx, images[i], reconstruction[i], out_tif, out_png)
            img_idx += 1


def main() -> None:
    """CLI entry point for VAE inference."""
    args = parse_args()
    device = init_device_and_seed(args.seed)

    config, autoencoder = load_config_and_model(args.config_file, args.checkpoint, device)
    print(f"[INFO] Loaded config from {args.config_file}")

    output_dir, out_tif, out_png = resolve_output_dirs(config.run_dir, args.input_dir, args.output_dir)
    print(f"[INFO] Output directory: {output_dir}")

    dataloader, image_paths = create_inference_dataloader(
        input_dir=args.input_dir,
        patch_size=tuple(config.autoencoder_train["patch_size"]),
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    num_images = len(image_paths)
    print(f"[INFO] Found {num_images} images in {args.input_dir}")
    print(f"[INFO] Loaded checkpoint from {args.checkpoint}")

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
