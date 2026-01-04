import argparse
import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from torch.amp import GradScaler, autocast
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
from pti_ldm_vae.utils.visualization import normalize_batch_for_display

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


def compute_scale_factor(
    vae: torch.nn.Module,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> float:
    """Compute a scale factor for latent normalization.

    Args:
        vae (torch.nn.Module): Frozen VAE model.
        sample_batch (tuple[torch.Tensor, torch.Tensor]): Batch of (edentulous, dentate) images.
        device (torch.device): Target device for computation.

    Returns:
        float: Scale factor defined as 1 / std(z).
    """
    images, _ = sample_batch
    images = images.to(device)
    amp_dtype = torch.float16 if device.type == "cuda" else None
    with torch.no_grad():
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            latent = vae.encode_stage_2_inputs(images)
    latent_std = torch.std(latent).item()
    if latent_std <= 0:
        return 1.0
    return 1.0 / latent_std


def log_sanity_samples(
    wandb_run: Any,
    *,
    unet: torch.nn.Module,
    vae: torch.nn.Module,
    regressor: Any,
    condition_builder: ConditionContextBuilder,
    metric_embed: MetricConditioning,
    schedule: DiffusionSchedule,
    concat_dentate: bool,
    scale_factor: float,
    batch: tuple[torch.Tensor, torch.Tensor],
    batch_idx: int,
    num_steps: int,
    step_indices: list[int],
    max_samples: int,
    epoch: int,
) -> None:
    """Log step-wise samples to W&B for visual sanity checks.

    Args:
        wandb_run (Any): Active W&B run or ``None``.
        unet (torch.nn.Module): Diffusion UNet.
        vae (torch.nn.Module): Frozen VAE for encoding/decoding.
        regressor (Any): Frozen regression head callable.
        condition_builder (ConditionContextBuilder): Conditioning module for latents.
        metric_embed (MetricConditioning): Conditioning module for metrics.
        schedule (DiffusionSchedule): Diffusion schedule.
        concat_dentate (bool): Whether to concatenate dentate latents to the UNet input.
        scale_factor (float): Latent scale factor (1 / std).
        batch (tuple[torch.Tensor, torch.Tensor]): Batch of (edentulous, dentate) images.
        batch_idx (int): Validation batch index.
        num_steps (int): Number of diffusion steps for sampling.
        step_indices (list[int]): 1-based sampling steps to visualize.
        max_samples (int): Max number of samples to log per batch.
        epoch (int): Current epoch index.
    """
    if wandb_run is None:
        return

    try:
        import wandb  # type: ignore
    except ImportError:
        return

    step_set = {int(step) for step in step_indices if 1 <= int(step) <= num_steps}
    if not step_set:
        return

    edente, dentate = batch
    device = next(unet.parameters()).device
    edente = edente.to(device)
    dentate = dentate.to(device)

    with torch.no_grad():
        z_cond = vae.encode_deterministic(dentate) * scale_factor
        metrics = regressor(dentate)
        metric_tokens = metric_embed(metrics)
        context = condition_builder(z_cond, metric_tokens)

        latent = torch.randn_like(z_cond)
        timesteps = torch.linspace(
            schedule.alphas.shape[0] - 1, 0, num_steps, device=device, dtype=torch.long
        ).long()

        saved_latents: dict[int, torch.Tensor] = {}
        for step_idx, t in enumerate(timesteps, start=1):
            latent_input = torch.cat([latent, z_cond], dim=1) if concat_dentate else latent
            timestep_batch = t.unsqueeze(0).repeat(latent.shape[0])
            eps = unet(latent_input, timesteps=timestep_batch, context=context)
            latent = schedule.step(eps, int(t.item()), latent, eta=0.0)
            if step_idx in step_set:
                saved_latents[step_idx] = latent.detach().clone()

        disp_dentate = normalize_batch_for_display(dentate.detach().cpu())
        disp_target = normalize_batch_for_display(edente.detach().cpu())

        for step_idx in sorted(saved_latents):
            decoded = vae.decode_stage_2_outputs(saved_latents[step_idx] / scale_factor)
            disp_generated = normalize_batch_for_display(decoded.detach().cpu())
            images = []
            for idx in range(min(max_samples, disp_dentate.shape[0])):
                triplet = torch.cat(
                    [disp_dentate[idx], disp_target[idx], disp_generated[idx]],
                    dim=2,
                )[0].numpy()
                images.append(wandb.Image(triplet, caption=f"step={step_idx} sample={idx:02d}"))

            if images:
                wandb_run.log(
                    {f"val/sample_epoch{epoch:03d}/batch_{batch_idx:03d}": images, "epoch": epoch + 1},
                    step=epoch + 1,
                )


def init_wandb(wandb_cfg: dict[str, Any], run_dir: Path, train_cfg: dict[str, Any], config_path: str):
    """Initialize Weights & Biases if enabled in config."""
    if not wandb_cfg.get("enabled", False):
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

    first_batch = next(iter(train_loader))
    scale_factor = compute_scale_factor(vae, first_batch, device)
    print(f"[INFO] Latent scale_factor: {scale_factor:.6f}")

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
        scale_factor=scale_factor,
        clip_grad=train_cfg.get("clip_grad"),
        ema_unet=ema_unet,
        ema_decay=train_cfg.get("ema_decay"),
    )

    state = TrainerState(epoch=0, global_step=0, best_val_loss=float("inf"))
    max_epochs = train_cfg["max_epochs"]
    val_interval = train_cfg.get("val_interval", 1)
    sanity_cfg = config.get("sanity_sampling", {})
    sanity_enabled = bool(sanity_cfg.get("enabled", True))
    sanity_every = int(sanity_cfg.get("every", 20))
    sanity_steps = int(sanity_cfg.get("num_steps", 50))
    sanity_step_indices = sanity_cfg.get("step_indices", [1, 10, 20, 40, 50])
    sanity_max_samples = int(sanity_cfg.get("max_samples", 1))
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

            if sanity_enabled and wandb_run is not None and sanity_every > 0 and epoch % sanity_every == 0:
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        log_sanity_samples(
                            wandb_run,
                            unet=trainer.unet,
                            vae=trainer.vae,
                            regressor=trainer.regressor,
                            condition_builder=trainer.condition_builder,
                            metric_embed=trainer.metric_embed,
                            schedule=schedule,
                            concat_dentate=concat_dentate,
                            scale_factor=scale_factor,
                            batch=batch,
                            batch_idx=batch_idx,
                            num_steps=sanity_steps,
                            step_indices=list(sanity_step_indices),
                            max_samples=sanity_max_samples,
                            epoch=epoch,
                        )

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
