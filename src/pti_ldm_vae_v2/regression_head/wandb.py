from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def init_wandb(
    config: dict[str, Any],
    run_dir: Path,
    latent_dim: int,
    targets: list[str],
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> Any:
    """Initialize a Weights & Biases run for regression training.

    Args:
        config (dict[str, Any]): Full regression configuration.
        run_dir (Path): Output run directory.
        latent_dim (int): Flattened latent dimension.
        targets (list[str]): Target names.
        data_cfg (dict[str, Any]): Data configuration block.
        train_cfg (dict[str, Any]): Training configuration block.

    Returns:
        Any: ``wandb.run`` instance when enabled, otherwise ``None``.
    """
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        print("[WARN] W&B is enabled but package 'wandb' is not installed.")
        return None

    project = os.getenv("WANDB_PROJECT", wandb_cfg.get("project", "pti-ldm_old-vae"))
    entity = wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
    run_name = wandb_cfg.get("name") or run_dir.name
    tags = wandb_cfg.get("tags", [])
    notes = wandb_cfg.get("notes", "")

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        notes=notes,
        dir=str(run_dir),
        config={
            "architecture": "vae-latent-regressor",
            "latent_dim": latent_dim,
            "targets": targets,
            "hidden_dims": config.get("regressor_def", {}).get("hidden_dims", []),
            "dropout": config.get("regressor_def", {}).get("dropout", 0.0),
            "activation": config.get("regressor_def", {}).get("activation", "relu"),
            "data_source": data_cfg.get("data_source", "edente"),
            "patch_size": data_cfg.get("patch_size"),
            "batch_size": train_cfg.get("batch_size"),
            "lr": train_cfg.get("lr"),
            "max_epochs": train_cfg.get("max_epochs"),
            "target_norm": train_cfg.get("target_norm", "none"),
            "loss": train_cfg.get("loss", "mse"),
        },
    )

    try:
        run.config.update({"full_config_json": config}, allow_val_change=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Could not attach full config to W&B: {exc}")

    print(f"W&B run initialized: {run.url}")
    return run


def log_regression_epoch(
    wandb_run: Any,
    *,
    epoch: int,
    train_loss: float,
    best_val: float,
    val_loss: float | None = None,
    metrics: dict[str, float] | None = None,
    loss_key: str = "loss_mse",
) -> None:
    """Log regression metrics to W&B if enabled.

    Args:
        wandb_run (Any): Active wandb run or ``None``.
        epoch (int): Current epoch index.
        train_loss (float): Training loss.
        best_val (float): Best validation loss so far.
        val_loss (float | None): Validation loss if computed this epoch.
        metrics (dict[str, float] | None): Additional metrics (MAE/MSE/R2).
        loss_key (str): Loss key used for logging.
    """
    if wandb_run is None:
        return

    payload: dict[str, Any] = {
        "epoch": epoch,
        f"train/{loss_key}": train_loss,
        f"best/val_{loss_key}": best_val,
    }
    if val_loss is not None:
        payload[f"val/{loss_key}"] = val_loss
    if metrics:
        payload.update({f"val/{k}": v for k, v in metrics.items()})
    try:
        wandb_run.log(payload)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to log to W&B: {exc}")
