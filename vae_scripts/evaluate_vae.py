import argparse
import json
import logging
import sys
from typing import Any

import numpy as np
import torch
from monai.losses import PerceptualLoss
from tqdm import tqdm

from pti_ldm_vae.models import VAEModel
from pti_ldm_vae.models.losses import compute_kl_loss
from pti_ldm_vae.utils import ensure_three_channels
from pti_ldm_vae.utils.cli_common import (
    add_shared_io_args,
    build_inference_dataloader,
    init_device_and_seed,
    load_config_and_model,
    resolve_eval_output_dir,
)
from pti_ldm_vae.utils.eval_metrics import compute_psnr, compute_ssim, serialize_args


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for VAE evaluation.

    Returns:
        argparse.Namespace: Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained VAE on a test set.")
    add_shared_io_args(parser, output_help="Output directory for metrics (default: evals/<config_name>/)")
    return parser.parse_args()


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
            reconstruction, z_mu, z_logvar = autoencoder(images)

        recon_clamped = torch.clamp(reconstruction, 0.0, 1.0)
        images_clamped = torch.clamp(images, 0.0, 1.0)

        intensity_val = intensity_loss_fn(reconstruction, images)
        kl_val = compute_kl_loss(z_mu, z_logvar)
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
    device = init_device_and_seed(args.seed)

    config, autoencoder = load_config_and_model(args.config_file, args.checkpoint, device)
    output_dir = resolve_eval_output_dir(args.config_file, args.output_dir)

    dataloader, image_paths = build_inference_dataloader(
        input_dir=args.input_dir,
        config=config,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )
    print(f"[INFO] Found {len(image_paths)} images in {args.input_dir}")
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
