from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def init_wandb(
    config: dict[str, Any],
    *,
    run_dir: Path,
    train_cfg: dict[str, Any],
    config_path: str,
) -> Any | None:
    """Initialize Weights & Biases logging.

    Args:
        config (dict[str, Any]): Full LDM configuration dictionary.
        run_dir (Path): Run directory used for local W&B files.
        train_cfg (dict[str, Any]): Training configuration block.
        config_path (str): Path to the config JSON file.

    Returns:
        Any | None: Active W&B run, or None when disabled/unavailable.
    """
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", True):
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        print("[WARN] W&B enabled but package 'wandb' is not installed.")
        return None

    project = os.getenv("WANDB_PROJECT", wandb_cfg.get("project", "pti-ldm-vae"))
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
            "architecture": "latent-diffusion",
            "batch_size": train_cfg.get("batch_size"),
            "lr": train_cfg.get("lr"),
            "max_epochs": train_cfg.get("max_epochs"),
            "clip_grad": train_cfg.get("clip_grad"),
            "ema_decay": train_cfg.get("ema_decay"),
            "config_file": config_path,
        },
    )

    try:
        with open(config_path, encoding="utf-8") as handle:
            full_cfg = handle.read()
        wandb.config.update({"full_config_json": full_cfg}, allow_val_change=True)
    except Exception as exc:
        print(f"[WARN] Could not attach full config to W&B: {exc}")

    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("time/epoch", step_metric="epoch")

    print(f"W&B run initialized: {run.url}")
    return run
