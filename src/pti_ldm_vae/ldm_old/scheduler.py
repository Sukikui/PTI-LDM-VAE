from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffusionSchedule:
    """Precomputed diffusion coefficients for forward and reverse steps."""

    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    @classmethod
    def linear(cls, timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> DiffusionSchedule:
        """Create a linear beta schedule.

        Args:
            timesteps: Number of diffusion steps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            device: Target torch device.

        Returns:
            DiffusionSchedule with precomputed coefficients.
        """
        betas = torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return cls(
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0 - alphas_cumprod),
        )

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward diffuse a batch of latents.

        Args:
            clean: Clean latent tensor.
            noise: Noise tensor (same shape as clean).
            timesteps: Timestep tensor [B].

        Returns:
            Noisy latent tensor.
        """
        sqrt_alpha_bar = self._gather(self.sqrt_alphas_cumprod, timesteps, clean)
        sqrt_one_minus = self._gather(self.sqrt_one_minus_alphas_cumprod, timesteps, clean)
        return sqrt_alpha_bar * clean + sqrt_one_minus * noise

    def step(
        self,
        model_pred: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Perform a DDIM-like update step.

        Args:
            model_pred (torch.Tensor): Predicted noise tensor.
            timestep (int): Current timestep index.
            sample (torch.Tensor): Current noisy latent.
            eta (float): DDIM noise scale (0 = deterministic).

        Returns:
            torch.Tensor: Updated latent tensor.
        """
        t = torch.tensor(timestep, device=sample.device, dtype=torch.long)
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=sample.device)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_prev = torch.sqrt(1.0 - alpha_bar_prev)

        pred_x0 = (sample - torch.sqrt(1.0 - alpha_bar_t) * model_pred) / torch.sqrt(alpha_bar_t)
        dir_term = torch.sqrt(torch.clamp(alpha_bar_prev, min=1e-8)) * model_pred
        if eta > 0 and t > 0:
            beta = 1 - alpha_bar_prev / alpha_bar_t
            noise = torch.randn_like(sample)
            sigma = eta * torch.sqrt(beta)
            return sqrt_alpha_bar_prev * pred_x0 + dir_term + sigma * noise
        return sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_prev * model_pred

    def step_with_prev(
        self,
        model_pred: torch.Tensor,
        timestep: int,
        prev_timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Perform a DDIM-like update step using an explicit previous timestep.

        Args:
            model_pred (torch.Tensor): Predicted noise tensor.
            timestep (int): Current timestep index.
            prev_timestep (int): Previous timestep index (can be non-consecutive).
            sample (torch.Tensor): Current noisy latent.
            eta (float): DDIM noise scale (0 = deterministic).

        Returns:
            torch.Tensor: Updated latent tensor.
        """
        t = torch.tensor(timestep, device=sample.device, dtype=torch.long)
        t_prev = torch.tensor(prev_timestep, device=sample.device, dtype=torch.long)
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t_prev] if prev_timestep >= 0 else torch.tensor(1.0, device=sample.device)
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_prev = torch.sqrt(1.0 - alpha_bar_prev)

        pred_x0 = (sample - torch.sqrt(1.0 - alpha_bar_t) * model_pred) / torch.sqrt(alpha_bar_t)
        dir_term = torch.sqrt(torch.clamp(alpha_bar_prev, min=1e-8)) * model_pred
        if eta > 0 and prev_timestep >= 0:
            beta = 1 - alpha_bar_prev / alpha_bar_t
            noise = torch.randn_like(sample)
            sigma = eta * torch.sqrt(beta)
            return sqrt_alpha_bar_prev * pred_x0 + dir_term + sigma * noise
        return sqrt_alpha_bar_prev * pred_x0 + sqrt_one_minus_prev * model_pred

    @staticmethod
    def _gather(values: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Gather per-timestep coefficients and reshape to broadcast."""
        out = values.gather(-1, timesteps)
        while out.ndim < reference.ndim:
            out = out.unsqueeze(-1)
        return out
