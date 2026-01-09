from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def init_wandb(
    wandb_cfg: dict[str, Any],
    *,
    run_dir: Path,
    config_file: str,
    train_cfg: dict[str, Any],
    config: Any,
) -> Any | None:
    """Initialize Weights & Biases logging.

    Args:
        wandb_cfg (dict[str, Any]): W&B configuration block.
        run_dir (Path): Run directory used for local W&B files.
        config_file (str): Path to the config JSON file.
        train_cfg (dict[str, Any]): Training configuration block.
        config (Any): Parsed configuration object.

    Returns:
        Any | None: Active W&B run, or None when disabled/unavailable.
    """
    if not wandb_cfg.get("enabled", True):
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        print("[WARN] W&B enabled but package 'wandb' is not installed.")
        return None

    project = os.getenv("WANDB_PROJECT", wandb_cfg.get("project", "pti-ldm_old-vae"))
    entity = wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
    run_name = wandb_cfg.get("name") or Path(run_dir).name
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
            "architecture": "VAE",
            "spatial_dims": config.spatial_dims,
            "latent_channels": config.latent_channels,
            "channels": config.autoencoder_def["channels"],
            "num_res_blocks": config.autoencoder_def["num_res_blocks"],
            "batch_size": train_cfg.get("batch_size"),
            "lr": train_cfg.get("lr"),
            "max_epochs": train_cfg.get("max_epochs"),
            "kl_weight": train_cfg.get("kl_weight"),
            "perceptual_weight": train_cfg.get("perceptual_weight"),
            "recon_loss": train_cfg.get("recon_loss"),
            "adv_weight": train_cfg.get("adv_weight"),
            "data_source": config.data_source,
            "patch_size": train_cfg.get("patch_size"),
            "train_split": config.train_split,
        },
    )

    try:
        with open(config_file, encoding="utf-8") as cfg_file:
            full_cfg = cfg_file.read()
        wandb.config.update({"full_config_json": full_cfg}, allow_val_change=True)
        artifact = wandb.Artifact("vae-config", type="config")
        artifact.add_file(config_file)
        wandb.log_artifact(artifact)
    except Exception as exc:
        print(f"[WARN] Could not upload config file to W&B: {exc}")

    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("epoch")
    wandb.define_metric("time_per_epoch", step_metric="epoch")

    print(f"✨ W&B run initialized: {run.url}")
    return run
