from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from typing import Any

import torch
from dotenv import load_dotenv
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from pti_ldm_vae_v2.vae.visualization import normalize_batch_for_display
from pti_ldm_vae_v2.vae_regression_common import DEFAULT_NUM_WORKERS, init_device_and_seed

from .build import build_frozen_regressor, build_frozen_vae, build_unet
from .conditioning import ConditionContextBuilder, MetricConditioning
from .config import apply_train_overrides, load_config, resolve_run_dir, resolve_run_dirs
from .data import create_ldm_dataloaders
from .noise import create_initial_latent, read_noise_init_config
from .scheduler import DiffusionSchedule
from .trainer import LDMTrainer, TrainerState
from .wandb import init_wandb

load_dotenv()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LDM training.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a latent diffusion model conditioned on dentate latents.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to LDM JSON config.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    return parser.parse_args()


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
    use_dentate_latent: bool,
    init_mode: str,
    noise_top: float,
    noise_bottom: float,
    noise_exponent: float,
    noise_direction: str,
    noise_weight: float,
    scale_factor: float,
    batch: tuple[torch.Tensor, torch.Tensor],
    batch_idx: int,
    num_steps: int,
    step_indices: list[int],
    max_samples: int,
    epoch: int,
    log_step: int,
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
        use_dentate_latent (bool): Whether to include dentate latents in cross-attention context.
        init_mode (str): ``pure_noise`` or ``dentate_noisy`` initialization.
        noise_top (float): Noise scale at the top of the image.
        noise_bottom (float): Noise scale at the bottom of the image.
        noise_exponent (float): Exponent to shape the vertical noise gradient.
        noise_direction (str): ``vertical`` or ``horizontal`` gradient direction.
        noise_weight (float): Global noise multiplier.
        scale_factor (float): Latent scale factor (1 / std).
        batch (tuple[torch.Tensor, torch.Tensor]): Batch of (edentulous, dentate) images.
        batch_idx (int): Validation batch index.
        num_steps (int): Number of diffusion steps for sampling.
        step_indices (list[int]): 1-based sampling steps to visualize.
        max_samples (int): Max number of samples to log per batch.
        epoch (int): Current epoch index.
        log_step (int): Global training step for W&B logging.
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
        if use_dentate_latent:
            context = condition_builder(z_cond, metric_tokens)
        else:
            context = metric_tokens.unsqueeze(1)

        latent = create_initial_latent(
            z_cond,
            init_mode=init_mode,
            noise_top=noise_top,
            noise_bottom=noise_bottom,
            noise_exponent=noise_exponent,
            noise_direction=noise_direction,
            noise_weight=noise_weight,
        )
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
                    step=log_step,
                )


def train() -> None:
    """Entry point for training the latent diffusion model."""
    args = parse_args()
    config = load_config(args.config_file)

    data_cfg = dict(config.get("data", {}))
    train_cfg = apply_train_overrides(
        config.get("train", {}),
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
    )
    conditioning_cfg = dict(config.get("conditioning", {}))
    diffusion_cfg = dict(config.get("diffusion", {}))
    unet_cfg = dict(config.get("unet", {}))
    noise_init = read_noise_init_config(config)

    run_dir = resolve_run_dir(config, args.config_file)
    _, weights_dir, splits_dir = resolve_run_dirs(run_dir)

    seed = data_cfg.get("seed", config.get("seed"))
    device = init_device_and_seed(seed, print_monai_config=False)
    wandb_run = init_wandb(config, run_dir=run_dir, train_cfg=train_cfg, config_path=args.config_file)

    train_loader, val_loader, train_pairs, val_pairs = create_ldm_dataloaders(
        data_base_dir=data_cfg["data_base_dir"],
        batch_size=train_cfg["batch_size"],
        patch_size=tuple(data_cfg["patch_size"]),
        train_split=float(data_cfg.get("train_split", 0.9)),
        num_workers=int(data_cfg.get("num_workers", DEFAULT_NUM_WORKERS)),
        seed=seed,
        subset_size=data_cfg.get("subset_size"),
        val_dir=data_cfg.get("val_dir"),
        cache_rate=float(data_cfg.get("cache_rate", 0.0)),
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
        patch_size=tuple(data_cfg["patch_size"]),
        targets=config["regressor"]["targets"],
    )

    use_dentate_latent = bool(conditioning_cfg.get("use_dentate_latent", True))
    concat_dentate = bool(conditioning_cfg.get("concat_dentate", True))
    if not use_dentate_latent and concat_dentate:
        print("[INFO] use_dentate_latent is False; forcing concat_dentate to False.")
        concat_dentate = False
    unet_config = dict(unet_cfg)
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
        use_dentate_latent=use_dentate_latent,
        noise_init_mode=str(noise_init["init_mode"]),
        noise_top=float(noise_init["noise_top"]),
        noise_bottom=float(noise_init["noise_bottom"]),
        noise_exponent=float(noise_init["noise_exponent"]),
        noise_direction=str(noise_init["noise_direction"]),
        noise_weight=float(noise_init["noise_weight"]),
        drop_z_prob=conditioning_cfg.get("condition_dropout", 0.0),
        drop_metrics_prob=conditioning_cfg.get("metrics_dropout", 0.0),
        scale_factor=scale_factor,
        clip_grad=train_cfg.get("clip_grad"),
        ema_unet=ema_unet,
        ema_decay=train_cfg.get("ema_decay"),
    )

    state = TrainerState(epoch=0, global_step=0, best_val_loss=float("inf"))
    max_epochs = int(train_cfg["max_epochs"])
    val_interval = int(train_cfg.get("val_interval", 1))
    sanity_cfg = dict(config.get("sanity_sampling", {}))
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
                            use_dentate_latent=use_dentate_latent,
                            init_mode=str(noise_init["init_mode"]),
                            noise_top=float(noise_init["noise_top"]),
                            noise_bottom=float(noise_init["noise_bottom"]),
                            noise_exponent=float(noise_init["noise_exponent"]),
                            noise_direction=str(noise_init["noise_direction"]),
                            noise_weight=float(noise_init["noise_weight"]),
                            scale_factor=scale_factor,
                            batch=batch,
                            batch_idx=batch_idx,
                            num_steps=sanity_steps,
                            step_indices=list(sanity_step_indices),
                            max_samples=sanity_max_samples,
                            epoch=epoch,
                            log_step=state.global_step,
                        )

        epoch_time = time.time() - epoch_start
        if wandb_run is not None:
            wandb_run.log({"time/epoch": epoch_time, "epoch": epoch + 1})

    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "ldm_pairs.json", "w", encoding="utf-8") as handle:
        json.dump({"train": train_pairs, "val": val_pairs}, handle, indent=2)
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    train()
