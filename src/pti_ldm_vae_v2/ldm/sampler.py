from __future__ import annotations

from collections.abc import Callable

import torch
from monai.networks.schedulers import DDIMScheduler

from pti_ldm_vae_v2.models.conditioning import CondEnc, ContextBuilder, apply_condition_dropout
from .noise import create_initial_latent


class LatentDiffusionSampler:
    """DDIM sampler for latent diffusion with metric + spatial conditioning.

    Args:
        unet (torch.nn.Module): Trained diffusion UNet.
        vae (torch.nn.Module): Frozen VAE model.
        condition_builder (ContextBuilder): Projects dentate latents to attention tokens.
        metric_embed (CondEnc): Embeds metric vectors to tokens.
        ddim_scheduler (DDIMScheduler): DDIM scheduler configured for sampling.
        concat_dentate (bool): Whether to concatenate dentate latents to UNet input channels.
        use_dentate_latent (bool): Whether to include dentate latents in cross-attention context.
        scale_factor (float): Latent scaling factor used during training.
    """

    def __init__(
        self,
        unet: torch.nn.Module,
        vae: torch.nn.Module,
        condition_builder: ContextBuilder,
        metric_embed: CondEnc,
        ddim_scheduler: DDIMScheduler,
        *,
        concat_dentate: bool,
        use_dentate_latent: bool,
        scale_factor: float = 1.0,
    ) -> None:
        self.unet = unet
        self.vae = vae
        self.condition_builder = condition_builder
        self.metric_embed = metric_embed
        self.ddim_scheduler = ddim_scheduler
        self.concat_dentate = concat_dentate
        self.use_dentate_latent = use_dentate_latent
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
        init_mode: str = "pure_noise",
        noise_top: float = 1.0,
        noise_bottom: float = 0.0,
        noise_exponent: float = 1.0,
        noise_direction: str = "vertical",
        noise_weight: float = 1.0,
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
            init_mode (str): ``pure_noise`` or ``dentate_noisy`` initialization.
            noise_top (float): Noise scale at the top of the image.
            noise_bottom (float): Noise scale at the bottom of the image.
            noise_exponent (float): Exponent to shape the vertical noise gradient.
            noise_direction (str): ``vertical`` or ``horizontal`` gradient direction.
            noise_weight (float): Global noise multiplier.

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
        z_cond_base = z_cond
        metrics = regressor(dentate_images)

        z_cond, metrics = apply_condition_dropout(
            z_cond,
            metrics,
            drop_z_prob,
            drop_metrics_prob,
            lambda shape: torch.rand(shape, device=device),
        )
        metric_tokens = self.metric_embed(metrics)
        if self.use_dentate_latent:
            context = self.condition_builder(z_cond, metric_tokens)
        else:
            context = metric_tokens.unsqueeze(1)

        self.ddim_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.ddim_scheduler.timesteps
        latent = create_initial_latent(
            z_cond_base,
            init_mode=init_mode,
            noise_top=noise_top,
            noise_bottom=noise_bottom,
            noise_exponent=noise_exponent,
            noise_direction=noise_direction,
            noise_weight=noise_weight,
        )

        for t in timesteps:
            latent_input = torch.cat([latent, z_cond], dim=1) if self.concat_dentate else latent
            timestep_batch = t.unsqueeze(0).repeat(latent.shape[0])
            eps = self.unet(latent_input, timesteps=timestep_batch, context=context)
            if guidance_scale is not None and guidance_scale > 1.0:
                z_zero, metrics_uncond = apply_condition_dropout(
                    z_cond,
                    metrics,
                    drop_z_prob=1.0,
                    drop_metrics_prob=1.0,
                    sampler=lambda shape: torch.zeros(shape, device=device),
                )
                metric_tokens_uncond = self.metric_embed(metrics_uncond)
                if self.use_dentate_latent:
                    context_uncond = self.condition_builder(z_zero, metric_tokens_uncond)
                else:
                    context_uncond = metric_tokens_uncond.unsqueeze(1)
                latent_input_uncond = torch.cat([latent, z_zero], dim=1) if self.concat_dentate else latent
                eps_uncond = self.unet(latent_input_uncond, timesteps=timestep_batch, context=context_uncond)
                eps = eps_uncond + guidance_scale * (eps - eps_uncond)
            latent, _ = self.ddim_scheduler.step(eps, int(t.item()), latent, eta=eta)
        return self.vae.decode_stage_2_outputs(latent / self.scale_factor)
