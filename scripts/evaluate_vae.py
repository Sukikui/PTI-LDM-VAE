import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from monai.config import print_config
from monai.losses import PerceptualLoss
from monai.utils import set_determinism
from tqdm import tqdm

from pti_ldm_vae.data import create_vae_inference_dataloader
from pti_ldm_vae.models import VAEModel
from pti_ldm_vae.models.losses import compute_kl_loss
from pti_ldm_vae.utils import ensure_three_channels
from pti_ldm_vae.utils.eval_metrics import compute_psnr, compute_ssim, serialize_args
from pti_ldm_vae.utils.vae_loader import default_eval_output_dir, load_vae_config, load_vae_model


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VAE evaluation.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE on a test set.")
    parser.add_argument("-c", "--config-file", required=True, help="Config json file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint (e.g., checkpoint_epoch73.pth)"
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input TIF images")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for metrics (default: evals/<config_name>/)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers (default: 4)",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def setup_output_dir(args: argparse.Namespace) -> Path:
    """Create the output directory used to store evaluation artifacts.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        Path: Concrete path where metrics and metadata are written.
    """
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = default_eval_output_dir(args.config_file)

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_dataloader(
    input_dir: str, patch_size: tuple[int, int], batch_size: int, num_samples: int | None, num_workers: int
) -> tuple[torch.utils.data.DataLoader, list[str]]:
    """Create dataloader for evaluation using shared VAE inference pipeline.

    Args:
        input_dir: Folder containing .tif images.
        patch_size: Resize target (H, W).
        batch_size: Batch size.
        num_samples: Optional limit on samples.
        num_workers: Number of dataloader workers.

    Returns:
        Tuple of dataloader and list of image paths.
    """
    return create_vae_inference_dataloader(
        input_dir=input_dir,
        patch_size=patch_size,
        batch_size=batch_size,
        num_samples=num_samples,
        num_workers=num_workers,
    )


def load_model(config: Any, checkpoint_path: str, device: torch.device) -> VAEModel:
    """Load VAE model from checkpoint.

    Args:
        config: Parsed config object.
        checkpoint_path: Path to checkpoint.
        device: Torch device.

    Returns:
        Loaded VAE model in eval mode.
    """
    autoencoder = VAEModel.from_config(config.autoencoder_def).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("autoencoder_state_dict", checkpoint)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    return autoencoder


def select_intensity_loss(config: Any) -> torch.nn.Module:
    """Select intensity loss (L1 or L2) to mirror training.

    Args:
        config: Parsed config object.

    Returns:
        Torch loss module.
    """
    if config.autoencoder_train.get("recon_loss") == "l2":
        return torch.nn.MSELoss()
    return torch.nn.L1Loss()


def evaluate(
    autoencoder: VAEModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    intensity_loss_fn: torch.nn.Module,
    perceptual_loss_fn: torch.nn.Module,
    perceptual_weight: float,
) -> dict[str, float]:
    """Run evaluation and compute aggregate metrics.

    Args:
        autoencoder: Loaded VAE model.
        dataloader: Evaluation dataloader.
        device: Torch device.
        intensity_loss_fn: Intensity loss module.
        perceptual_loss_fn: Perceptual loss module.
        perceptual_weight: Weight applied to perceptual loss.

    Returns:
        Dictionary of aggregated metric means/stds.
    """
    metrics: dict[str, list[float]] = {
        "recon_loss": [],
        "kl_loss": [],
        "perceptual_loss": [],
        "psnr": [],
        "ssim": [],
        "loss_total": [],
        "mse": [],
        "mae": [],
    }

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch.to(device)
        with torch.no_grad():
            reconstruction, z_mu, z_sigma = autoencoder(images)

        recon_clamped = torch.clamp(reconstruction, 0.0, 1.0)
        images_clamped = torch.clamp(images, 0.0, 1.0)

        intensity_val = intensity_loss_fn(reconstruction, images)
        kl_val = compute_kl_loss(z_mu, z_sigma)
        perc_val = perceptual_loss_fn(
            ensure_three_channels(reconstruction.float()), ensure_three_channels(images.float())
        )
        psnr_val = compute_psnr(recon_clamped, images_clamped)
        ssim_val = compute_ssim(recon_clamped, images_clamped)
        mse_val = torch.mean((recon_clamped - images_clamped) ** 2, dim=(1, 2, 3))
        mae_val = torch.mean(torch.abs(recon_clamped - images_clamped), dim=(1, 2, 3))

        total_val = intensity_val + kl_val + perceptual_weight * perc_val

        metrics["recon_loss"].append(float(intensity_val.item()))
        metrics["kl_loss"].append(float(kl_val.item()))
        metrics["perceptual_loss"].append(float(perc_val.item()))
        metrics["psnr"].extend(psnr_val.cpu().tolist())
        metrics["ssim"].extend(ssim_val.cpu().tolist())
        metrics["loss_total"].append(float(total_val.item()))
        metrics["mse"].extend(mse_val.cpu().tolist())
        metrics["mae"].extend(mae_val.cpu().tolist())

    summary: dict[str, float] = {}
    for key, values in metrics.items():
        if len(values) == 0:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def save_metrics(output_dir: Path, summary: dict[str, float], image_paths: list[str], args: argparse.Namespace) -> None:
    """Persist metrics and evaluated filenames.

    Args:
        output_dir (Path): Folder to write JSON.
        summary (Dict[str, float]): Aggregated metrics.
        image_paths (List[str]): List of evaluated file paths.
        args (argparse.Namespace): Parsed CLI arguments to record provenance.
    """
    payload = {"args": serialize_args(args), "metrics": summary, "files": image_paths}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print_config()
    set_determinism(args.seed)

    config = load_vae_config(args.config_file)
    output_dir = setup_output_dir(args)
    patch_size = tuple(config.autoencoder_train["patch_size"])

    dataloader, image_paths = create_dataloader(
        input_dir=args.input_dir,
        patch_size=patch_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Found {len(image_paths)} images in {args.input_dir}")

    autoencoder = load_vae_model(config, args.checkpoint, device)
    intensity_loss_fn = select_intensity_loss(config)
    perceptual_loss_fn = PerceptualLoss(spatial_dims=config.spatial_dims, network_type="squeeze").to(device)

    summary = evaluate(
        autoencoder=autoencoder,
        dataloader=dataloader,
        device=device,
        intensity_loss_fn=intensity_loss_fn,
        perceptual_loss_fn=perceptual_loss_fn,
        perceptual_weight=config.autoencoder_train["perceptual_weight"],
    )
    save_metrics(output_dir, summary, image_paths, args)

    print("\n=== Evaluation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    print(f"\nMetrics saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
