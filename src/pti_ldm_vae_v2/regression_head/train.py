from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from torch.optim import Adam
from tqdm import tqdm

from .checkpoint import (
    load_regression_checkpoint,
    maybe_save_best_regression_checkpoint,
    save_last_regression_checkpoint,
)
from .config import (
    build_regression_model_from_config,
    extract_regression_data_config,
    extract_regression_train_config,
    extract_regressor_def_config,
    load_config,
    resolve_run_dir,
)
from .data import create_regression_dataloaders
from .engine import train_one_epoch, validate_one_epoch
from .losses import build_loss_fn, regression_loss_key
from .normalization import NORM_STATS_FILENAME, TargetNormalizer, compute_target_normalizer, save_target_normalizer
from pti_ldm_vae_v2.common import init_device_and_seed
from .wandb import init_wandb, log_regression_epoch

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for regression head training.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Train a regression head on frozen VAE latents.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to regression config JSON.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Optional checkpoint to resume the head.")
    return parser.parse_args()


def apply_overrides(
    config: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply CLI overrides and return normalized regression config sections.

    Args:
        config (dict[str, Any]): Loaded JSON config.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: Data, training, and regressor configuration blocks.
    """
    data_cfg = extract_regression_data_config(config)
    train_cfg = extract_regression_train_config(config)
    regressor_cfg = extract_regressor_def_config(config)

    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.max_epochs is not None:
        train_cfg["max_epochs"] = args.max_epochs
    if args.seed is not None:
        data_cfg["seed"] = args.seed

    config["data"] = data_cfg
    config["regression_train"] = train_cfg
    config["regressor_def"] = regressor_cfg
    return data_cfg, train_cfg, regressor_cfg


def summarize_model(model: Any, latent_dim: int, targets: list[str], reg_cfg: dict[str, Any]) -> None:
    """Print a compact summary of the regression model.

    Args:
        model (Any): Regression wrapper (VAE + MLP head).
        latent_dim (int): Flattened latent dimension.
        targets (list[str]): Target names predicted by the head.
        reg_cfg (dict[str, Any]): Regressor configuration.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nModel summary")
    print("-" * 60)
    print(f"Flattened latent dim: {latent_dim}")
    print(f"Regressor hidden dims: {reg_cfg.get('hidden_dims', [])}")
    print(f"Activation: {reg_cfg.get('activation', 'relu')} | Dropout: {reg_cfg.get('dropout', 0.0)}")
    print(f"Targets: {targets} (#{len(targets)})")
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")
    print("-" * 60)


def maybe_build_normalizer(
    dataset: object, target_names: list[str], weights_dir: Path, target_norm: str
) -> TargetNormalizer | None:
    """Compute and save target normalizer if requested.

    Args:
        dataset (object): Dataset exposing ``stacked_targets``.
        target_names (list[str]): Ordered target names.
        weights_dir (Path): Directory to persist stats.
        target_norm (str): Normalization mode.

    Returns:
        TargetNormalizer | None: Normalizer if created, else ``None``.
    """
    if target_norm.lower() != "standard":
        return None
    if not hasattr(dataset, "stacked_targets"):
        raise ValueError("Dataset must expose stacked_targets() to compute normalization statistics.")

    stacked = dataset.stacked_targets()
    normalizer = compute_target_normalizer(stacked)
    save_target_normalizer(weights_dir / NORM_STATS_FILENAME, normalizer, target_names)
    return normalizer


def train() -> None:
    """Entry point for training the regression head."""
    args = parse_args()
    config = load_config(args.config_file)
    data_cfg, train_cfg, _ = apply_overrides(config, args)

    run_dir = resolve_run_dir(config, args.config_file)
    weights_dir = run_dir / "trained_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    seed = data_cfg.get("seed")
    device = init_device_and_seed(seed)

    targets: list[str] = config["targets"]
    model, latent_dim = build_regression_model_from_config(config, targets, device)
    summarize_model(model, latent_dim, targets, config["regressor_def"])
    wandb_run = init_wandb(config, run_dir, latent_dim, targets, data_cfg, train_cfg)

    train_loader, val_loader, train_paths, val_paths = create_regression_dataloaders(
        data_base_dir=data_cfg["data_base_dir"],
        attributes_path=data_cfg["attributes_path"],
        targets=targets,
        batch_size=train_cfg["batch_size"],
        patch_size=tuple(data_cfg["patch_size"]),
        train_split=float(data_cfg.get("train_split", 0.9)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        seed=seed,
        subset_size=data_cfg.get("subset_size"),
        val_dir=data_cfg.get("val_dir"),
        cache_rate=float(data_cfg.get("cache_rate", 0.0)),
        data_source=data_cfg.get("data_source", "edente"),
        normalize_attributes=data_cfg.get("normalize_attributes"),
    )

    normalizer = maybe_build_normalizer(
        train_loader.dataset, targets, weights_dir, train_cfg.get("target_norm", "none")
    )
    loss_fn = build_loss_fn(train_cfg.get("loss", "mse"))
    loss_key = regression_loss_key(train_cfg)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    if args.resume_checkpoint is not None:
        load_regression_checkpoint(Path(args.resume_checkpoint), model, targets)

    max_epochs = int(train_cfg["max_epochs"])
    val_interval = int(train_cfg.get("val_interval", 1))
    best_val = float("inf")
    best_checkpoint_path: Path | None = None
    epoch_iter = tqdm(range(1, max_epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, normalizer)
        log_prefix = f"[Epoch {epoch:03d}/{max_epochs:03d}]"

        if epoch % val_interval == 0 or epoch == max_epochs:
            val_loss, metrics = validate_one_epoch(model, val_loader, loss_fn, device, targets, normalizer)
            best_val, best_checkpoint_path = maybe_save_best_regression_checkpoint(
                weights_dir=weights_dir,
                model=model,
                targets=targets,
                epoch=epoch,
                val_loss=val_loss,
                best_val_loss=best_val,
                best_path=best_checkpoint_path,
            )
            epoch_iter.set_postfix(**{loss_key: f"{train_loss:.4f}", f"val_{loss_key}": f"{val_loss:.4f}"})
            tqdm.write(
                f"{log_prefix} train_{loss_key}={train_loss:.4f} val_{loss_key}={val_loss:.4f} metrics={metrics}"
            )
            log_regression_epoch(
                wandb_run,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
                best_val=best_val,
                loss_key=loss_key,
            )
        else:
            epoch_iter.set_postfix(**{loss_key: f"{train_loss:.4f}", f"best_val_{loss_key}": f"{best_val:.4f}"})
            tqdm.write(f"{log_prefix} train_{loss_key}={train_loss:.4f}")
            log_regression_epoch(
                wandb_run,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=None,
                metrics=None,
                best_val=best_val,
                loss_key=loss_key,
            )

        save_last_regression_checkpoint(weights_dir, model, targets, epoch)

    print("Training complete")
    print(f"   Trained on {len(train_paths)} images, validated on {len(val_paths)}")
    print(f"   Weights: {weights_dir}")
    if best_val < float("inf") and best_checkpoint_path is not None:
        print(f"   Best checkpoint: {best_checkpoint_path} (val_{loss_key}={best_val:.4f})")
    if normalizer is not None:
        print(f"   Normalization stats: {weights_dir / NORM_STATS_FILENAME}")
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


def main() -> None:
    """CLI entry point for regression head training."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()


if __name__ == "__main__":
    main()
