from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from monai.losses import PerceptualLoss
from tqdm import tqdm
from typing import Any

from .config import default_eval_output_dir, load_config_and_model
from .data import create_inference_dataloader
from .losses import compute_kl_loss, ensure_three_channels, select_intensity_loss
from pti_ldm_vae_v2.vae_regression_common import VAEModel
from pti_ldm_vae_v2.vae_regression_common import init_device_and_seed


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VAE evaluation.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE on a test set.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Directory with input TIF images.")
    parser.add_argument("--output-dir", default=None, help="Output directory for metrics JSON.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to process.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    return parser.parse_args()


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """Compute PSNR for a batch.

    Args:
        pred (torch.Tensor): Predicted images [B, C, H, W].
        target (torch.Tensor): Reference images [B, C, H, W].
        data_range (float): Value range for pixels.

    Returns:
        torch.Tensor: PSNR per sample.
    """
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    return 10 * torch.log10(torch.tensor(data_range, device=pred.device) ** 2 / mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute SSIM for single-channel images using a Gaussian window.

    Args:
        pred (torch.Tensor): Predicted images [B, C, H, W].
        target (torch.Tensor): Reference images [B, C, H, W].
        data_range (float): Value range for pixels.
        k1 (float): Stabilization constant K1.
        k2 (float): Stabilization constant K2.

    Returns:
        torch.Tensor: SSIM per sample.
    """
    window_size = 11
    sigma = 1.5
    coords = torch.arange(window_size, device=pred.device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    kernel_1d = g
    kernel_2d = (kernel_1d.t() @ kernel_1d).unsqueeze(0).unsqueeze(0)
    pad = window_size // 2

    def _filter(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, kernel_2d, padding=pad, groups=x.shape[1])

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    mu_x = _filter(pred)
    mu_y = _filter(target)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _filter(pred * pred) - mu_x2
    sigma_y2 = _filter(target * target) - mu_y2
    sigma_xy = _filter(pred * target) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
    return ssim_map.mean(dim=(1, 2, 3))


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI arguments to JSON-serializable primitives.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        dict[str, Any]: Mapping of argument names to primitive values.
    """
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        if hasattr(value, "__fspath__"):
            serialized[key] = str(value)
        elif isinstance(value, (list, tuple)):
            serialized[key] = [str(item) for item in value]
        else:
            serialized[key] = value
    return serialized


def evaluate(
    *,
    autoencoder: VAEModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    intensity_loss_fn: torch.nn.Module,
    perceptual_loss_fn: torch.nn.Module,
    perceptual_weight: float,
) -> dict[str, float]:
    """Run evaluation and compute aggregate metrics.

    Args:
        autoencoder (VAEModel): Loaded VAE model.
        dataloader (torch.utils.data.DataLoader): Evaluation dataloader.
        device (torch.device): Torch device.
        intensity_loss_fn (torch.nn.Module): Intensity loss module.
        perceptual_loss_fn (torch.nn.Module): Perceptual loss module.
        perceptual_weight (float): Weight applied to perceptual loss.

    Returns:
        dict[str, float]: Aggregated metrics summary.
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
            reconstruction, z_mu, z_logvar = autoencoder(images)

        recon_clamped = torch.clamp(reconstruction, 0.0, 1.0)
        images_clamped = torch.clamp(images, 0.0, 1.0)

        intensity_val = intensity_loss_fn(reconstruction, images)
        kl_val = compute_kl_loss(z_mu, z_logvar)
        perc_val = perceptual_loss_fn(
            ensure_three_channels(reconstruction.float()),
            ensure_three_channels(images.float()),
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
        summary (dict[str, float]): Aggregated metrics.
        image_paths (list[str]): List of evaluated file paths.
        args (argparse.Namespace): Parsed CLI arguments to record provenance.
    """
    payload = {"args": serialize_args(args), "metrics": summary, "files": image_paths}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    """CLI entry point for VAE evaluation."""
    args = parse_args()
    device = init_device_and_seed(args.seed)

    config, autoencoder = load_config_and_model(args.config_file, args.checkpoint, device)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_eval_output_dir(run_dir=config.run_dir, input_dir=args.input_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader, image_paths = create_inference_dataloader(
        input_dir=args.input_dir,
        patch_size=tuple(config.autoencoder_train["patch_size"]),
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    print(f"[INFO] Found {len(image_paths)} images in {args.input_dir}")

    intensity_loss_fn = select_intensity_loss(config.autoencoder_train.get("recon_loss"))
    perceptual_loss_fn = PerceptualLoss(spatial_dims=config.spatial_dims, network_type="squeeze").to(device)

    summary = evaluate(
        autoencoder=autoencoder,
        dataloader=dataloader,
        device=device,
        intensity_loss_fn=intensity_loss_fn,
        perceptual_loss_fn=perceptual_loss_fn,
        perceptual_weight=float(config.autoencoder_train["perceptual_weight"]),
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
