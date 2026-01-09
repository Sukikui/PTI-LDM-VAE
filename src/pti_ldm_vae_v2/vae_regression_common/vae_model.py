from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import AutoencoderKL


class VAEModel(nn.Module):
    """Variational Autoencoder wrapper around MONAI's AutoencoderKL.

    This is a thin wrapper that simplifies configuration and instantiation
    while exposing all MONAI AutoencoderKL functionality.

    Encoding modes:
        - encode_stage_2_inputs(): Stochastic sampling (z = z_mu + eps * sigma).
          Use this for training diffusion models (Stage 2).

        - encode_deterministic(): Deterministic encoding using z_mu only.
          Use this for inference, analysis, and visualization.

    Args:
        spatial_dims (int): Number of spatial dimensions (2 for 2D, 3 for 3D).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        latent_channels (int): Number of channels in latent space.
        channels (list[int]): Channel dimensions for each resolution level.
        num_res_blocks (int): Number of residual blocks per resolution level.
        norm_num_groups (int): Number of groups for group normalization.
        norm_eps (float): Epsilon for numerical stability in normalization.
        attention_levels (list[bool] | None): Attention flags for each level.
        with_encoder_nonlocal_attn (bool): Enable non-local attention in encoder.
        with_decoder_nonlocal_attn (bool): Enable non-local attention in decoder.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        channels: list[int],
        num_res_blocks: int = 2,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        attention_levels: list[bool] | None = None,
        with_encoder_nonlocal_attn: bool = True,
        with_decoder_nonlocal_attn: bool = True,
    ) -> None:
        super().__init__()

        if attention_levels is None:
            attention_levels = [False] * len(channels)

        self.autoencoder = AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            channels=channels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            attention_levels=attention_levels,
            with_encoder_nonlocal_attn=with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=with_decoder_nonlocal_attn,
        )

    @classmethod
    def from_config(cls, config: dict) -> "VAEModel":
        """Create a VAEModel from a configuration dictionary.

        Args:
            config (dict): Dictionary containing model configuration parameters.

        Returns:
            VAEModel: Initialized VAEModel instance.
        """
        return cls(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            latent_channels=config["latent_channels"],
            channels=config["channels"],
            num_res_blocks=config.get("num_res_blocks", 2),
            norm_num_groups=config.get("norm_num_groups", 32),
            norm_eps=config.get("norm_eps", 1e-6),
            attention_levels=config.get("attention_levels"),
            with_encoder_nonlocal_attn=config.get("with_encoder_nonlocal_attn", True),
            with_decoder_nonlocal_attn=config.get("with_decoder_nonlocal_attn", True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstruction, z_mu, z_logvar.
        """
        return self.autoencoder(x)

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs for stage 2 (diffusion model training).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sampled latent tensor.
        """
        return self.autoencoder.encode_stage_2_inputs(x)

    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs deterministically using the latent mean.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Latent tensor (mean of distribution, deterministic).
        """
        z_mu, _ = self.autoencoder.encode(x)
        return z_mu

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        """Decode outputs from stage 2 (diffusion model inference).

        Args:
            z (torch.Tensor): Latent tensor from diffusion model.

        Returns:
            torch.Tensor: Decoded image tensor.
        """
        return self.autoencoder.decode_stage_2_outputs(z)

    def reconstruct_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct inputs using deterministic latent mean.

        Args:
            x (torch.Tensor): Input tensor to reconstruct.

        Returns:
            torch.Tensor: Reconstructed tensor decoded from ``z_mu``.
        """
        z_mu = self.encode_deterministic(x)
        return self.decode_stage_2_outputs(z_mu)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load state dict into the autoencoder.

        Args:
            state_dict (dict): State dictionary to load.
            strict (bool): Whether to enforce strict key matching.
        """
        self.autoencoder.load_state_dict(state_dict, strict=strict)

    def state_dict(self) -> dict:
        """Get the state dict of the autoencoder.

        Returns:
            dict: Autoencoder state dictionary.
        """
        return self.autoencoder.state_dict()
