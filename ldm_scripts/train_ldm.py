import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from torch.amp import GradScaler
from torch.optim import AdamW
from tqdm import tqdm

from pti_ldm_vae.ldm import (
    ConditionContextBuilder,
    DiffusionSchedule,
    LDMTrainer,
    MetricConditioning,
    TrainerState,
    build_frozen_regressor,
    build_frozen_vae,
    build_unet,
    create_ldm_dataloaders,
)
from pti_ldm_vae.utils.cli_common import load_json_config

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Train a latent diffusion model conditioned on dentate latents.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to LDM JSON config.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    return parser.parse_args()


def set_seed(seed: int | None) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Optional seed value.
    """
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_device() -> torch.device:
    """Select CUDA if available, otherwise CPU.

    Returns:
        Selected torch device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def override_train_cfg(train_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply CLI overrides to training configuration.

    Args:
        train_cfg: Base training configuration.
        args: Parsed CLI arguments.

    Returns:
        Training configuration with overrides applied.
    """
    if args.max_epochs is not None:
        train_cfg["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    return train_cfg


def init_wandb(wandb_cfg: dict[str, Any], run_dir: Path, train_cfg: dict[str, Any], config_path: str):
    """Initialize Weights & Biases if enabled in config."""
    if not wandb_cfg.get("enabled", False):
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[WARN] W&B enabled but package 'wandb' is not installed.")
        return None

    project = wandb_cfg.get("project") or os.getenv("WANDB_PROJECT", "pti-ldm-vae")
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
            "lr": train_cfg.get("lr"),
            "batch_size": train_cfg.get("batch_size"),
            "max_epochs": train_cfg.get("max_epochs"),
            "clip_grad": train_cfg.get("clip_grad"),
            "ema_decay": train_cfg.get("ema_decay"),
            "config_file": config_path,
        },
    )
    try:
        run.config.update({"full_config_json": load_json_config(config_path)}, allow_val_change=True)
    except Exception as exc:
        print(f"[WARN] Could not attach full config to W&B: {exc}")
    return run


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config_file)

    device = prepare_device()
    set_seed(config.get("seed", 42))

    data_cfg = config.get("data", {})
    train_cfg = override_train_cfg(config.get("train", {}), args)
    conditioning_cfg = config.get("conditioning", {})
    diffusion_cfg = config.get("diffusion", {})
    unet_cfg = config.get("unet", {})
    run_dir = Path(config.get("run_dir", "runs/ldm_run"))
    weights_dir = run_dir / "trained_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(config.get("wandb", {}), run_dir, train_cfg, args.config_file)

    patch_size = tuple(data_cfg["patch_size"])
    train_loader, val_loader, train_pairs, val_pairs = create_ldm_dataloaders(
        data_base_dir=data_cfg["data_base_dir"],
        batch_size=train_cfg["batch_size"],
        patch_size=patch_size,
        train_split=float(data_cfg.get("train_split", 0.9)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        seed=data_cfg.get("seed", config.get("seed", 42)),
        subset_size=data_cfg.get("subset_size"),
        val_dir=data_cfg.get("val_dir"),
        cache_rate=float(data_cfg.get("cache_rate", 0.0)),
        distributed=False,
    )

    vae, latent_channels = build_frozen_vae(
        config_file=config["vae"]["config_file"],
        checkpoint=config["vae"]["checkpoint"],
        device=device,
    )
    regressor = build_frozen_regressor(
        config_file=config["regressor"]["config_file"],
        checkpoint=config["regressor"]["checkpoint"],
        device=device,
        patch_size=patch_size,
        targets=config["regressor"]["targets"],
    )

    concat_dentate = bool(conditioning_cfg.get("concat_dentate", True))
    unet_config = deepcopy(unet_cfg)
    unet_config["in_channels"] = latent_channels * (2 if concat_dentate else 1)
    unet_config["out_channels"] = latent_channels

    unet = build_unet(unet_config).to(device)
    ema_unet = deepcopy(unet).to(device) if train_cfg.get("ema_decay") else None

    cross_attention_dim = unet_config.get("cross_attention_dim", 256)
    metric_embed = MetricConditioning(
        input_dim=len(config["regressor"]["targets"]),
        embed_dim=cross_attention_dim,
        dropout=conditioning_cfg.get("metric_dropout", 0.0),
    ).to(device)
    condition_builder = ConditionContextBuilder(
        latent_channels=latent_channels,
        cross_attention_dim=cross_attention_dim,
    ).to(device)

    schedule = DiffusionSchedule.linear(
        timesteps=diffusion_cfg.get("num_train_timesteps", 1000),
        beta_start=diffusion_cfg.get("beta_start", 0.00085),
        beta_end=diffusion_cfg.get("beta_end", 0.012),
        device=device,
    )

    optimizer = AdamW(
        list(unet.parameters()) + list(metric_embed.parameters()) + list(condition_builder.parameters()),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    trainer = LDMTrainer(
        unet=unet,
        vae=vae,
        regressor=regressor,
        condition_builder=condition_builder,
        metric_embed=metric_embed,
        schedule=schedule,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        concat_dentate=concat_dentate,
        drop_z_prob=conditioning_cfg.get("condition_dropout", 0.0),
        drop_metrics_prob=conditioning_cfg.get("metrics_dropout", 0.0),
        clip_grad=train_cfg.get("clip_grad"),
        ema_unet=ema_unet,
        ema_decay=train_cfg.get("ema_decay"),
    )

    state = TrainerState(epoch=0, global_step=0, best_val_loss=float("inf"))
    max_epochs = train_cfg["max_epochs"]
    val_interval = train_cfg.get("val_interval", 1)
    for epoch in range(max_epochs):
        epoch_start = time.time()
        trainer.unet.train()
        trainer.metric_embed.train()
        trainer.condition_builder.train()
        train_loss_sum = 0.0
        train_steps = 0
        for batch in tqdm(train_loader, desc=f"Train {epoch + 1}/{max_epochs}", unit="batch"):
            loss = trainer.training_step(batch)
            state.global_step += 1
            train_loss_sum += loss.item()
            train_steps += 1
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss_total": loss.item(),
                        "train/noise_loss": loss.item(),
                        "train/step": state.global_step,
                    }
                )

        train_loss = train_loss_sum / max(train_steps, 1)
        print(f"[Epoch {epoch + 1}/{max_epochs}] train_loss={train_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss_total_epoch": train_loss,
                    "train/noise_loss_epoch": train_loss,
                    "epoch": epoch + 1,
                }
            )

        state.epoch = epoch
        run_validation = (epoch + 1) % val_interval == 0 or epoch == max_epochs - 1
        if run_validation:
            trainer.unet.eval()
            trainer.metric_embed.eval()
            trainer.condition_builder.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val {epoch + 1}/{max_epochs}", unit="batch"):
                    val_loss = trainer.validation_step(batch)
                    val_loss_sum += val_loss.item()
                    val_steps += 1
            val_loss = val_loss_sum / max(val_steps, 1)
            print(f"[Epoch {epoch + 1}/{max_epochs}] val_loss={val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "val/loss_total": val_loss,
                        "val/noise_loss": val_loss,
                        "epoch": epoch + 1,
                    }
                )
            trainer.save_checkpoint(state, weights_dir, best=False)
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                trainer.save_checkpoint(state, weights_dir, best=True)
                if wandb_run is not None:
                    wandb_run.summary["best/val_loss_total"] = val_loss

        epoch_time = time.time() - epoch_start
        if wandb_run is not None:
            wandb_run.log({"time/epoch": epoch_time, "epoch": epoch + 1})

    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "ldm_pairs.json", "w", encoding="utf-8") as handle:
        json.dump({"train": train_pairs, "val": val_pairs}, handle, indent=2)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
