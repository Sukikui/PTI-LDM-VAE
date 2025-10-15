import torch
import torch.nn as nn
from monai.networks.nets import AutoencoderKL


class VAEModel(nn.Module):
    """Variational Autoencoder wrapper around MONAI's AutoencoderKL.

    This is a thin wrapper that simplifies configuration and instantiation
    while exposing all MONAI AutoencoderKL functionality.

    Args:
        spatial_dims: Number of spatial dimensions (2 for 2D, 3 for 3D)
        in_channels: Number of input channels
        out_channels: Number of output channels
        latent_channels: Number of channels in latent space
        channels: List of channel dimensions for each resolution level
        num_res_blocks: Number of residual blocks per resolution level
        norm_num_groups: Number of groups for group normalization
        norm_eps: Epsilon for numerical stability in normalization
        attention_levels: List of booleans indicating which levels use attention
        with_encoder_nonlocal_attn: Whether to use non-local attention in encoder
        with_decoder_nonlocal_attn: Whether to use non-local attention in decoder

    Example:
        >>> config = {"spatial_dims": 2, "in_channels": 1, ...}
        >>> vae = VAEModel.from_config(config)
        >>> reconstruction, z_mu, z_sigma = vae(images)
        >>> z = vae.encode_stage_2_inputs(images)
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
            config: Dictionary containing model configuration parameters

        Returns:
            Initialized VAEModel instance
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
            x: Input tensor

        Returns:
            Tuple of (reconstruction, z_mu, z_sigma)
        """
        return self.autoencoder(x)

    def encode_stage_2_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs for stage 2 (diffusion model training).

        Args:
            x: Input tensor

        Returns:
            Sampled latent tensor
        """
        return self.autoencoder.encode_stage_2_inputs(x)

    def decode_stage_2_outputs(self, z: torch.Tensor) -> torch.Tensor:
        """Decode outputs from stage 2 (diffusion model inference).

        Args:
            z: Latent tensor from diffusion model

        Returns:
            Decoded image tensor
        """
        return self.autoencoder.decode_stage_2_outputs(z)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load state dict into the autoencoder."""
        self.autoencoder.load_state_dict(state_dict, strict=strict)

    def state_dict(self) -> dict:
        """Get the state dict of the autoencoder."""
        return self.autoencoder.state_dict()
