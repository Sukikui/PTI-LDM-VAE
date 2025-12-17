import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from pti_ldm_vae.models import VAEModel
from pti_ldm_vae.utils.cli_common import (
    add_shared_io_args,
    build_inference_dataloader,
    init_device_and_seed,
    load_config_and_model,
    resolve_inference_output_dirs,
)
from pti_ldm_vae.utils.visualization import normalize_batch_for_display


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="VAE Inference Script")
    add_shared_io_args(parser, output_help="Output directory (default: inference_<checkpoint_name>)")
    return parser.parse_args()


def save_results(idx: int, input_img: torch.Tensor, recon_img: torch.Tensor, out_tif: Path, out_png: Path) -> None:
    """Save a single result as TIF and PNG.

    Args:
        idx (int): Global image index for naming outputs.
        input_img (torch.Tensor): Original input image tensor without batch dimension.
        recon_img (torch.Tensor): Reconstructed image tensor without batch dimension.
        out_tif (Path): Destination directory for TIF outputs.
        out_png (Path): Destination directory for PNG outputs.
    """
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
    """Run inference and save results.

    Args:
        autoencoder (VAEModel): Loaded VAE model.
        dataloader (DataLoader): Dataloader yielding images to reconstruct.
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

        # Save each image in batch
        for i in range(images.shape[0]):
            save_results(img_idx, images[i], reconstruction[i], out_tif, out_png)
            img_idx += 1


def main() -> None:
    args = parse_args()
    device = init_device_and_seed(args.seed)

    # Load configuration and model
    config, autoencoder = load_config_and_model(args.config_file, args.checkpoint, device)
    print(f"[INFO] Loaded config from {args.config_file}")

    # Setup output directories
    output_dir, out_tif, out_png = resolve_inference_output_dirs(args.checkpoint, args.output_dir)
    print(f"[INFO] Output directory: {output_dir}")

    # Create dataloader
    dataloader, image_paths = build_inference_dataloader(
        input_dir=args.input_dir,
        config=config,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )
    num_images = len(image_paths)
    print(f"[INFO] Found {num_images} images in {args.input_dir}")
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
