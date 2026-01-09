from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class MetricConditioning(nn.Module):
    """Embed metric vectors for diffusion conditioning.

    Args:
        input_dim (int): Dimension of the raw metric vector.
        embed_dim (int): Target embedding dimension (cross-attention dim).
        dropout (float): Dropout probability applied to metrics.
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, metrics: torch.Tensor) -> torch.Tensor:
        """Embed metrics into conditioning tokens.

        Args:
            metrics (torch.Tensor): Tensor of shape [B, A].

        Returns:
            torch.Tensor: Embedded tensor of shape [B, embed_dim].
        """
        metrics = self.dropout(metrics)
        return self.mlp(metrics)


class ConditionContextBuilder(nn.Module):
    """Project spatial latents and metric embeddings to cross-attention context.

    Args:
        latent_channels (int): Number of channels in dentate latents.
        cross_attention_dim (int): Target dimension for attention tokens.
    """

    def __init__(self, latent_channels: int, cross_attention_dim: int) -> None:
        super().__init__()
        self.project_latent = nn.Linear(latent_channels, cross_attention_dim)

    def forward(self, latent: torch.Tensor, metric_tokens: torch.Tensor | None = None) -> torch.Tensor:
        """Create context tokens for cross-attention.

        Args:
            latent (torch.Tensor): Tensor [B, C, H, W] representing dentate latents.
            metric_tokens (torch.Tensor | None): Optional tensor [B, D] already embedded.

        Returns:
            torch.Tensor: Context tokens shaped [B, N_tokens, cross_attention_dim].
        """
        batch_size, channels, height, width = latent.shape
        spatial_tokens = latent.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        spatial_tokens = self.project_latent(spatial_tokens)
        if metric_tokens is None:
            return spatial_tokens
        metric_tokens = metric_tokens.unsqueeze(1)
        return torch.cat([spatial_tokens, metric_tokens], dim=1)


def apply_condition_dropout(
    z_dentate: torch.Tensor,
    metrics: torch.Tensor,
    drop_z_prob: float,
    drop_metrics_prob: float,
    sampler: Callable[[torch.Size], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop conditioning signals for classifier-free guidance.

    Args:
        z_dentate (torch.Tensor): Dentate latent tensor [B, C, H, W].
        metrics (torch.Tensor): Metric tensor [B, A].
        drop_z_prob (float): Probability of zeroing dentate conditioning.
        drop_metrics_prob (float): Probability of zeroing metric conditioning.
        sampler (Callable[[torch.Size], torch.Tensor]): Random sampler producing values in [0, 1].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Possibly dropped (z_dentate, metrics).
    """
    batch = z_dentate.shape[0]
    if drop_z_prob > 0:
        mask = sampler((batch, 1, 1, 1)) < drop_z_prob
        z_dentate = z_dentate.masked_fill(mask, 0.0)
    if drop_metrics_prob > 0:
        mask = sampler((batch, 1)) < drop_metrics_prob
        metrics = metrics.masked_fill(mask, 0.0)
    return z_dentate, metrics
