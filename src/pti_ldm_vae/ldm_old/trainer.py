from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer

from pti_ldm_vae.ldm_old.conditioning import ConditionContextBuilder, MetricConditioning, apply_condition_dropout
from pti_ldm_vae.ldm_old.scheduler import DiffusionSchedule


@dataclass
class TrainerState:
    """Track training state for checkpointing."""

    epoch: int
    global_step: int
    best_val_loss: float


class LDMTrainer:
    """Latent diffusion training loop."""

    def __init__(
        self,
        *,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        regressor: Callable[[torch.Tensor], torch.Tensor],
        condition_builder: ConditionContextBuilder,
        metric_embed: MetricConditioning,
        schedule: DiffusionSchedule,
        optimizer: Optimizer,
        scaler: GradScaler,
        device: torch.device,
        concat_dentate: bool,
        drop_z_prob: float,
        drop_metrics_prob: float,
        scale_factor: float = 1.0,
        clip_grad: float | None = None,
        ema_unet: torch.nn.Module | None = None,
        ema_decay: float | None = 0.999,
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.regressor = regressor
        self.condition_builder = condition_builder
        self.metric_embed = metric_embed
        self.schedule = schedule
        self.optimizer = optimizer
        self.scaler = scaler
        self.device = device
        self.concat_dentate = concat_dentate
        self.drop_z_prob = drop_z_prob
        self.drop_metrics_prob = drop_metrics_prob
        self.scale_factor = float(scale_factor)
        self.clip_grad = clip_grad
        self.ema_unet = ema_unet
        self.ema_decay = ema_decay

    def _ema_update(self, decay: float) -> None:
        """Update EMA weights."""
        if self.ema_unet is None or self.ema_decay is None:
            return
        with torch.no_grad():
            for ema_param, param in zip(self.ema_unet.parameters(), self.unet.parameters(), strict=True):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Tuple of (edentulous images, dentate images).

        Returns:
            torch.Tensor: Detached loss tensor.
        """
        images, condition_images = batch
        images = images.to(self.device)
        condition_images = condition_images.to(self.device)

        with torch.no_grad():
            z_target = self.vae.encode_stage_2_inputs(images)
            z_condition = self.vae.encode_deterministic(condition_images)
            metrics = self.regressor(condition_images)

        z_target = z_target * self.scale_factor
        z_condition = z_condition * self.scale_factor

        z_condition, metrics = apply_condition_dropout(
            z_condition,
            metrics,
            self.drop_z_prob,
            self.drop_metrics_prob,
            lambda shape: torch.rand(shape, device=self.device),
        )
        metric_tokens = self.metric_embed(metrics)
        context = self.condition_builder(z_condition, metric_tokens)

        noise = torch.randn_like(z_target)
        timesteps = torch.randint(
            low=0,
            high=self.schedule.alphas_cumprod.shape[0],
            size=(images.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        noisy_latent = self.schedule.add_noise(z_target, noise, timesteps)
        unet_input = torch.cat([noisy_latent, z_condition], dim=1) if self.concat_dentate else noisy_latent

        self.optimizer.zero_grad(set_to_none=True)
        amp_dtype = torch.float16 if self.device.type == "cuda" else None
        with autocast(device_type=self.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            noise_pred = self.unet(unet_input, timesteps=timesteps, context=context)
            loss = F.mse_loss(noise_pred.float(), noise.float())

        self.scaler.scale(loss).backward()
        if self.clip_grad is not None:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.clip_grad)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.ema_decay is not None:
            self._ema_update(decay=self.ema_decay)
        return loss.detach()

    @torch.no_grad()
    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute validation loss on one batch.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): Tuple of (edentulous images, dentate images).

        Returns:
            torch.Tensor: Detached MSE loss tensor.
        """
        images, condition_images = batch
        images = images.to(self.device)
        condition_images = condition_images.to(self.device)

        z_target = self.vae.encode_deterministic(images)
        z_condition = self.vae.encode_deterministic(condition_images)
        metrics = self.regressor(condition_images)

        z_target = z_target * self.scale_factor
        z_condition = z_condition * self.scale_factor

        metric_tokens = self.metric_embed(metrics)
        context = self.condition_builder(z_condition, metric_tokens)

        noise = torch.randn_like(z_target)
        timesteps = torch.randint(
            low=0,
            high=self.schedule.alphas_cumprod.shape[0],
            size=(images.shape[0],),
            device=self.device,
            dtype=torch.long,
        )
        noisy_latent = self.schedule.add_noise(z_target, noise, timesteps)
        unet_input = torch.cat([noisy_latent, z_condition], dim=1) if self.concat_dentate else noisy_latent
        noise_pred = self.unet(unet_input, timesteps=timesteps, context=context)
        return F.mse_loss(noise_pred.float(), noise.float()).detach()

    def save_checkpoint(
        self,
        state: TrainerState,
        checkpoint_dir: Path,
        best: bool,
    ) -> Path:
        """Persist the UNet (and EMA if present) along with optimizer state.

        Args:
            state (TrainerState): Trainer state metadata.
            checkpoint_dir (Path): Destination directory.
            best (bool): Whether this is the best checkpoint.

        Returns:
            Path: Path to the saved file.
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        suffix = "best" if best else "last"
        path = checkpoint_dir / f"ldm_unet_{suffix}.pth"
        torch.save(
            {
                "epoch": state.epoch,
                "global_step": state.global_step,
                "best_val_loss": state.best_val_loss,
                "unet_state_dict": self.unet.state_dict(),
                "ema_unet_state_dict": self.ema_unet.state_dict() if self.ema_unet is not None else None,
                "metric_embed_state_dict": self.metric_embed.state_dict(),
                "condition_builder_state_dict": self.condition_builder.state_dict(),
                "scale_factor": self.scale_factor,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        return path
