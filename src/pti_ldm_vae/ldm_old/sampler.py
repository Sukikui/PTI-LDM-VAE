from __future__ import annotations

from collections.abc import Callable

import torch

from pti_ldm_vae.ldm_old.conditioning import ConditionContextBuilder, MetricConditioning, apply_condition_dropout
from pti_ldm_vae.ldm_old.scheduler import DiffusionSchedule


class LatentDiffusionSampler:
    """DDIM sampler for latent diffusion with metric + spatial conditioning.

    Args:
        unet: Trained diffusion UNet.
        vae: Frozen VAE model.
        condition_builder: Projects dentate latents to attention tokens.
        metric_embed: Embeds metric vectors to tokens.
        schedule: DiffusionSchedule instance.
        concat_dentate: When True, concatenate dentate latent to UNet input channels.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        condition_builder: ConditionContextBuilder,
        metric_embed: MetricConditioning,
        schedule: DiffusionSchedule,
        *,
        concat_dentate: bool,
        scale_factor: float = 1.0,
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.condition_builder = condition_builder
        self.metric_embed = metric_embed
        self.schedule = schedule
        self.concat_dentate = concat_dentate
        self.scale_factor = float(scale_factor)

    @torch.no_grad()
    def __call__(
        self,
        dentate_images: torch.Tensor,
        regressor: Callable[[torch.Tensor], torch.Tensor],
        num_steps: int,
        guidance_scale: float | None = None,
        drop_z_prob: float = 0.0,
        drop_metrics_prob: float = 0.0,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate edentulous reconstructions from dentate images.

        Args:
            dentate_images (torch.Tensor): Input dentate images [B, C, H, W].
            regressor (Callable[[torch.Tensor], torch.Tensor]): Callable predicting metrics from images.
            num_steps (int): Number of reverse diffusion steps.
            guidance_scale (float | None): Guidance strength (None disables CFG).
            drop_z_prob (float): Dropout prob. for dentate latent during sampling.
            drop_metrics_prob (float): Dropout prob. for metrics during sampling.
            eta (float): DDIM noise scale.

        Returns:
            torch.Tensor: Generated edentulous images decoded by the VAE.
        """
        device = dentate_images.device
        self.unet.eval()
        self.vae.eval()
        self.metric_embed.eval()
        self.condition_builder.eval()
        z_cond = self.vae.encode_deterministic(dentate_images)
        z_cond = z_cond * self.scale_factor
        metrics = regressor(dentate_images)

        rng = torch.Generator(device=device)
        z_cond, metrics = apply_condition_dropout(
            z_cond, metrics, drop_z_prob, drop_metrics_prob, lambda shape: torch.rand(shape, device=device)
        )
        metric_tokens = self.metric_embed(metrics)
        context = self.condition_builder(z_cond, metric_tokens)

        timesteps = torch.linspace(
            len(self.schedule.alphas) - 1,
            0,
            num_steps,
            device=device,
            dtype=torch.long,
        ).long()
        latent = torch.randn_like(z_cond)

        for idx, t in enumerate(timesteps):
            prev_t = timesteps[idx + 1] if idx + 1 < len(timesteps) else torch.tensor(-1, device=device)
            latent_input = torch.cat([latent, z_cond], dim=1) if self.concat_dentate else latent
            timestep_batch = t.unsqueeze(0).repeat(latent.shape[0])
            eps = self.unet(latent_input, timesteps=timestep_batch, context=context)
            if guidance_scale is not None and guidance_scale > 1.0:
                z_zero, metrics_uncond = apply_condition_dropout(
                    z_cond,
                    metrics,
                    drop_z_prob=1.0,
                    drop_metrics_prob=1.0,
                    sampler=lambda s: torch.zeros(s, device=device),
                )
                metric_tokens_uncond = self.metric_embed(metrics_uncond)
                context_uncond = self.condition_builder(z_zero, metric_tokens_uncond)
                latent_input_uncond = torch.cat([latent, z_zero], dim=1) if self.concat_dentate else latent
                eps_uncond = self.unet(latent_input_uncond, timesteps=timestep_batch, context=context_uncond)
                eps = eps_uncond + guidance_scale * (eps - eps_uncond)
            latent = self.schedule.step_with_prev(eps, int(t.item()), int(prev_t.item()), latent, eta=eta)
        return self.vae.decode_stage_2_outputs(latent / self.scale_factor)
