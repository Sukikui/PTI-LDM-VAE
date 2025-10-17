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
from monai.data import DataLoader, Dataset
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, LoadImage, Resize
from monai.utils import set_determinism
from PIL import Image
from tqdm import tqdm

from pti_ldm_vae.data import LocalNormalizeByMask
from pti_ldm_vae.models import VAEModel
from pti_ldm_vae.utils.visualization import normalize_batch_for_display


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VAE Inference Script")
    parser.add_argument("-c", "--config-file", default="./config/config_train_16g_cond.json", help="Config json file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., checkpoint_epoch73.pth)"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input TIF images")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (default: inference_<checkpoint_name>)"
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    return parser.parse_args()


def load_config(config_file: str) -> Any:
    """Load configuration from JSON file."""
    config_dict = json.load(open(config_file))

    class Config:
        pass

    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)

    return config


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
    input_dir: str, patch_size: tuple, batch_size: int, num_samples: int | None = None
) -> tuple[DataLoader, int]:
    """Create dataloader for inference."""
    input_path = Path(input_dir)
    image_paths = sorted(input_path.glob("*.tif"))

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No .tif images found in {input_dir}")

    # Limit number of samples if specified
    if num_samples is not None:
        image_paths = image_paths[:num_samples]

    # Define transforms
    transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(patch_size),
            LocalNormalizeByMask(),
            EnsureType(dtype=torch.float32),
        ]
    )

    # Create dataset and dataloader
    dataset = Dataset(data=image_paths, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return dataloader, len(image_paths)


def load_model(config: Any, checkpoint_path: str, device: torch.device) -> VAEModel:
    """Load VAE model from checkpoint."""
    autoencoder = VAEModel.from_config(config.autoencoder_def).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    autoencoder.eval()

    return autoencoder


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
            reconstruction, _, _ = autoencoder(images)

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
    dataloader, num_images = create_dataloader(args.input_dir, patch_size, args.batch_size, args.num_samples)
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
