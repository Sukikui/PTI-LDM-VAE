from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from .checkpoint import load_regression_checkpoint
from .config import (
    build_regression_model_from_config,
    extract_regression_data_config,
    extract_regression_eval_config,
    extract_regression_train_config,
    load_config,
    resolve_run_dir,
)
from .data import create_regression_eval_dataloader
from .engine import validate_one_epoch
from .losses import build_loss_fn
from .normalization import NORM_STATS_FILENAME, load_target_normalizer
from pti_ldm_vae_v2.common import init_device_and_seed, resolve_run_output_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for regression head evaluation.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate a regression head on VAE latents.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to regression config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint of the trained head.")
    parser.add_argument(
        "--input-dir",
        required=False,
        default=None,
        help="Directory containing validation/test images (default: evaluation.data_base_dir).",
    )
    parser.add_argument(
        "--attributes-path",
        required=False,
        default=None,
        help="Attributes JSON for evaluation targets (default: evaluation.attributes_path).",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to write metrics.json.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-samples", type=int, default=None, help="Evaluate only first N samples.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for determinism.")
    return parser.parse_args()


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
        else:
            serialized[key] = value
    return serialized


def save_metrics(output_dir: Path, metrics: dict[str, float], args: argparse.Namespace, files: list[str]) -> None:
    """Persist metrics to JSON.

    Args:
        output_dir (Path): Output directory for metrics.
        metrics (dict[str, float]): Metrics payload.
        args (argparse.Namespace): Parsed CLI arguments.
        files (list[str]): Evaluated file paths.
    """
    payload = {"metrics": metrics, "args": serialize_args(args), "files": files}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
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
        tuple[dict[str, Any], dict[str, Any]]: Data and evaluation configuration blocks.
    """
    data_cfg = extract_regression_data_config(config)
    train_cfg = extract_regression_train_config(config)
    eval_cfg = extract_regression_eval_config(config, data_cfg)

    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.attributes_path is not None:
        eval_cfg["attributes_path"] = args.attributes_path

    config["data"] = data_cfg
    config["regression_train"] = train_cfg
    config["evaluation"] = eval_cfg
    return train_cfg, eval_cfg


def evaluate() -> None:
    """Entry point for regression head evaluation."""
    args = parse_args()
    config = load_config(args.config_file)
    train_cfg, eval_cfg = normalize_configs(config, args)
    run_dir = resolve_run_dir(config, args.config_file)
    device = init_device_and_seed(args.seed)

    targets: list[str] = config["targets"]
    model, _ = build_regression_model_from_config(config, targets, device)
    load_regression_checkpoint(Path(args.checkpoint), model, targets)

    batch_size = train_cfg["batch_size"]
    input_dir = args.input_dir or eval_cfg["data_base_dir"]
    attributes_path = eval_cfg["attributes_path"]

    dataloader, image_paths = create_regression_eval_dataloader(
        input_dir=input_dir,
        attributes_path=attributes_path,
        targets=targets,
        patch_size=tuple(eval_cfg["patch_size"]),
        batch_size=batch_size,
        num_workers=int(eval_cfg.get("num_workers", 4)),
        num_samples=args.num_samples,
        data_source=eval_cfg.get("data_source", "edente"),
        normalize_attributes=eval_cfg.get("normalize_attributes"),
    )

    normalizer = load_optional_normalizer(run_dir, targets)
    loss_fn = build_loss_fn(train_cfg.get("loss", "mse"))
    val_loss, metrics = validate_one_epoch(model, dataloader, loss_fn, device, targets, normalizer)

    args_resolved = vars(args).copy()
    args_resolved["resolved_input_dir"] = input_dir
    args_resolved["resolved_attributes_path"] = attributes_path
    output_dir = resolve_run_output_dir(run_dir, input_dir, args.output_dir, "eval")
    save_metrics(output_dir, {"val_loss": val_loss, **metrics}, argparse.Namespace(**args_resolved), image_paths)
    print("Evaluation complete")
    print(f"   Metrics written to {output_dir / 'metrics.json'}")


def main() -> None:
    """CLI entry point for regression head evaluation."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    evaluate()


if __name__ == "__main__":
    main()
