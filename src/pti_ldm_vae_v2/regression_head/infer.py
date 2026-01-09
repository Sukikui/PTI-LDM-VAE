from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_regression_checkpoint
from .config import (
    build_regression_model_from_config,
    extract_regression_data_config,
    extract_regression_train_config,
    load_config,
    resolve_run_dir,
)
from .data import create_regression_inference_dataloader
from .normalization import NORM_STATS_FILENAME, load_target_normalizer
from pti_ldm_vae_v2.vae_regression_common import init_device_and_seed, resolve_run_output_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for regression head inference.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference with a regression head on VAE latents.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to regression config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint of the trained head.")
    parser.add_argument("--input-dir", required=True, help="Directory containing images.")
    parser.add_argument("--output-dir", default=None, help="Directory to write predictions.json.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of images.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for determinism.")
    return parser.parse_args()


def save_predictions(output_dir: Path, target_names: list[str], files: list[str], preds: torch.Tensor) -> None:
    """Persist predictions to JSON.

    Args:
        output_dir (Path): Output directory.
        target_names (list[str]): Ordered target names.
        files (list[str]): Image file paths.
        preds (torch.Tensor): Predictions tensor [B, T].
    """
    payload = {
        "predictions": {
            Path(path).name: {name: float(preds[idx, j].item()) for j, name in enumerate(target_names)}
            for idx, path in enumerate(files)
        }
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_optional_normalizer(run_dir: Path, target_names: list[str]) -> Any | None:
    """Load normalization stats if present.

    Args:
        run_dir (Path): Run directory.
        target_names (list[str]): Ordered target names.

    Returns:
        Any | None: Normalizer instance or None.
    """
    norm_path = run_dir / "trained_weights" / NORM_STATS_FILENAME
    if norm_path.exists():
        return load_target_normalizer(norm_path, target_names)
    return None


def normalize_configs(config: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return normalized data and training configs with CLI overrides.

    Args:
        config (dict[str, Any]): Loaded configuration dictionary.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Data and training configuration blocks.
    """
    data_cfg = extract_regression_data_config(config)
    train_cfg = extract_regression_train_config(config)

    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    config["data"] = data_cfg
    config["regression_train"] = train_cfg
    return data_cfg, train_cfg


def infer() -> None:
    """Entry point for regression head inference."""
    args = parse_args()
    config = load_config(args.config_file)
    data_cfg, train_cfg = normalize_configs(config, args)
    run_dir = resolve_run_dir(config, args.config_file)
    device = init_device_and_seed(args.seed)

    targets: list[str] = config["targets"]
    model, _ = build_regression_model_from_config(config, targets, device)
    load_regression_checkpoint(Path(args.checkpoint), model, targets)

    batch_size = train_cfg["batch_size"]
    dataloader, image_paths = create_regression_inference_dataloader(
        input_dir=args.input_dir,
        patch_size=tuple(data_cfg["patch_size"]),
        batch_size=batch_size,
        num_samples=args.num_samples,
        num_workers=int(data_cfg.get("num_workers", 4)),
    )

    normalizer = load_optional_normalizer(run_dir, targets)
    preds_all: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            preds = model(images)
            if normalizer is not None:
                preds = normalizer.denormalize(preds)
            preds_all.append(preds.cpu())

    stacked = torch.cat(preds_all, dim=0)
    output_dir = resolve_run_output_dir(run_dir, args.input_dir, args.output_dir, "inference")
    save_predictions(output_dir, targets, image_paths, stacked)
    print("Inference complete")
    print(f"   Predictions written to {output_dir / 'predictions.json'}")


def main() -> None:
    """CLI entry point for regression head inference."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    infer()


if __name__ == "__main__":
    main()
