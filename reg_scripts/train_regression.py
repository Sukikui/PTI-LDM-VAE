import argparse
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from torch.optim import Adam
from tqdm import tqdm

from pti_ldm_vae.data import create_regression_dataloaders
from pti_ldm_vae.utils.cli_common import init_device_and_seed, load_json_config, resolve_run_dir
from pti_ldm_vae.utils.regression_utils import (
    TargetNormalizer,
    build_loss_fn,
    build_regression_model_from_config,
    compute_target_normalizer,
    extract_regression_data_config,
    extract_regression_train_config,
    extract_regressor_def_config,
    init_regression_wandb,
    load_regression_checkpoint,
    log_regression_epoch,
    regression_loss_key,
    save_regression_checkpoint,
    save_target_normalizer,
    train_one_epoch,
    validate_one_epoch,
)

NORM_STATS_FILENAME = "target_norm_stats.json"

# Load environment variables from .env file (for WANDB_PROJECT, WANDB_ENTITY, etc.)
load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for regression head training."""
    parser = argparse.ArgumentParser(description="Train a regression head on frozen VAE latents.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to regression config JSON.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers.")
    parser.add_argument("--cache-rate", type=float, default=None, help="Override cache rate.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument("--subset-size", type=int, default=None, help="Use first N images for a quick run.")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Optional checkpoint to resume the head.")
    return parser.parse_args()


def apply_overrides(
    config: dict[str, object], args: argparse.Namespace
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Apply CLI overrides and return normalized regression config sections.

    Args:
        config (dict[str, object]): Loaded JSON config.
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        tuple[dict[str, object], dict[str, object], dict[str, object]]: Data, training, and regressor configuration blocks.
    """
    data_cfg = extract_regression_data_config(config)
    train_cfg = extract_regression_train_config(config)
    regressor_cfg = extract_regressor_def_config(config)

    data_overrides = {
        "num_workers": args.num_workers,
        "cache_rate": args.cache_rate,
        "seed": args.seed,
        "subset_size": args.subset_size,
    }
    overrides = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
    }
    for key, value in data_overrides.items():
        if value is not None:
            data_cfg[key] = value
    for key, value in overrides.items():
        if value is not None:
            train_cfg[key] = value

    config["data"] = data_cfg
    config["regression_train"] = train_cfg
    config["regressor_def"] = regressor_cfg
    return data_cfg, train_cfg, regressor_cfg


def summarize_model(model: torch.nn.Module, latent_dim: int, targets: list[str], reg_cfg: dict[str, Any]) -> None:
    """Print a compact summary of the regression model.

    Args:
        model (torch.nn.Module): Regression wrapper (VAE + MLP head).
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


def main() -> None:
    """Entry point for training the regression head."""
    args = parse_args()
    config = load_json_config(args.config_file)
    data_cfg, train_cfg, _ = apply_overrides(config, args)
    run_dir = resolve_run_dir(config, args.config_file)
    weights_dir = run_dir / "trained_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    seed = data_cfg.get("seed")
    device = init_device_and_seed(seed)

    targets: list[str] = config["targets"]
    model, latent_dim = build_regression_model_from_config(config, targets, device)
    summarize_model(model, latent_dim, targets, config["regressor_def"])
    wandb_run = init_regression_wandb(config, run_dir, latent_dim, targets, data_cfg, train_cfg)

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

    max_epochs = train_cfg["max_epochs"]
    val_interval = train_cfg.get("val_interval", 1)
    best_val = float("inf")
    epoch_iter = tqdm(range(1, max_epochs + 1), desc="Epochs", unit="epoch")
    for epoch in epoch_iter:
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, normalizer)
        log_prefix = f"[Epoch {epoch:03d}/{max_epochs:03d}]"

        if epoch % val_interval == 0 or epoch == max_epochs:
            val_loss, metrics = validate_one_epoch(model, val_loader, loss_fn, device, targets, normalizer)
            if val_loss < best_val:
                best_val = val_loss
                best_path = weights_dir / f"head_epoch{epoch:03d}.pth"
                save_regression_checkpoint(best_path, model, targets, epoch)
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

        last_path = weights_dir / "head_last.pth"
        save_regression_checkpoint(last_path, model, targets, epoch)

    print("✅ Training complete")
    print(f"   Trained on {len(train_paths)} images, validated on {len(val_paths)}")
    print(f"   Weights: {weights_dir}")
    if normalizer is not None:
        print(f"   Normalization stats: {weights_dir / NORM_STATS_FILENAME}")
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
