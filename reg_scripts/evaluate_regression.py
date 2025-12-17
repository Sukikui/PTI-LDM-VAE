import argparse
import json
from pathlib import Path
from typing import Any

from pti_ldm_vae.data import create_regression_eval_dataloader
from pti_ldm_vae.utils.cli_common import init_device_and_seed, load_json_config, resolve_run_dir
from pti_ldm_vae.utils.regression_utils import (
    build_loss_fn,
    build_regression_model_from_config,
    extract_regression_data_config,
    extract_regression_train_config,
    load_regression_checkpoint,
    load_target_normalizer,
    validate_one_epoch,
)

NORM_STATS_FILENAME = "target_norm_stats.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for regression head evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a regression head on VAE latents.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to regression config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint of the trained head.")
    parser.add_argument("--input-dir", required=True, help="Directory containing validation/test images.")
    parser.add_argument("--output-dir", default=None, help="Directory to write metrics.json (default: <run_dir>/eval).")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--num-samples", type=int, default=None, help="Evaluate only first N samples.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for determinism.")
    return parser.parse_args()


def save_metrics(output_dir: Path, metrics: dict[str, float], args: argparse.Namespace, files: list[str]) -> None:
    """Persist metrics to JSON."""
    payload = {"metrics": metrics, "args": vars(args), "files": files}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_optional_normalizer(run_dir: Path, target_names: list[str]):
    """Load normalization stats if present."""
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
    if args.num_workers is not None:
        data_cfg["num_workers"] = args.num_workers

    config["data"] = data_cfg
    config["regression_train"] = train_cfg
    return data_cfg, train_cfg


def main() -> None:
    """Entry point for regression head evaluation."""
    args = parse_args()
    config = load_json_config(args.config_file)
    data_cfg, train_cfg = normalize_configs(config, args)
    run_dir = resolve_run_dir(config, args.config_file)
    device = init_device_and_seed(args.seed)

    targets: list[str] = config["targets"]
    model, _ = build_regression_model_from_config(config, targets, device)
    load_regression_checkpoint(Path(args.checkpoint), model, targets)

    batch_size = train_cfg["batch_size"]
    num_workers = data_cfg.get("num_workers", 4)

    dataloader, image_paths = create_regression_eval_dataloader(
        input_dir=args.input_dir,
        attributes_path=data_cfg["attributes_path"],
        targets=targets,
        patch_size=tuple(data_cfg["patch_size"]),
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=args.num_samples,
        data_source=data_cfg.get("data_source", "edente"),
        normalize_attributes=data_cfg.get("normalize_attributes"),
    )

    normalizer = load_optional_normalizer(run_dir, targets)
    loss_fn = build_loss_fn(train_cfg.get("loss", "mse"))
    val_loss, metrics = validate_one_epoch(model, dataloader, loss_fn, device, targets, normalizer)

    output_dir = Path(args.output_dir) if args.output_dir is not None else run_dir / "eval"
    save_metrics(output_dir, {"val_loss": val_loss, **metrics}, args, image_paths)
    print("✅ Evaluation complete")
    print(f"   Metrics written to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
