import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet


def create_condition_projector(
    condition_input_dim: int,
    cross_attention_dim: int,
):
    """Create a linear projection layer for conditioning.

    This projector is used to transform condition features (e.g., latent representations)
    to the cross-attention dimension expected by the UNet.

    Args:
        condition_input_dim: Input dimension (e.g., VAE latent channels = 4)
        cross_attention_dim: Output dimension for cross-attention

    Returns:
        Linear projection layer

    Example:
        >>> # In training script (line 114 of train_diffusion_tif_cond.py)
        >>> projector = create_condition_projector(4, 512)
        >>> # Later, in optimizer
        >>> optimizer = torch.optim.Adam(
        ...     list(unet.parameters()) + list(projector.parameters()),
        ...     lr=1e-5
        ... )
    """
    return nn.Linear(condition_input_dim, cross_attention_dim)


class DiffusionUNet(nn.Module):
    """Diffusion UNet wrapper around MONAI's DiffusionModelUNet.

    This is a thin wrapper that simplifies configuration and instantiation
    while exposing all MONAI DiffusionModelUNet functionality.

    Args:
        spatial_dims: Number of spatial dimensions (2 for 2D, 3 for 3D)
        in_channels: Number of input channels (should match VAE latent channels)
        out_channels: Number of output channels (usually same as in_channels)
        channels: List of channel dimensions for each resolution level
        attention_levels: List of booleans indicating which levels use attention
        num_head_channels: List of number of channels per attention head at each level
        num_res_blocks: Number of residual blocks per resolution level
        with_conditioning: Whether to use cross-attention conditioning
        cross_attention_dim: Dimension for cross-attention features
        norm_num_groups: Number of groups for group normalization

    Example:
        >>> config = {
        ...     "spatial_dims": 2,
        ...     "in_channels": 4,
        ...     "out_channels": 4,
        ...     "channels": [32, 64, 128, 256],
        ...     "attention_levels": [False, True, True, True],
        ...     "num_head_channels": [0, 32, 32, 32],
        ...     "with_conditioning": True,
        ...     "cross_attention_dim": 512,
        ... }
        >>> unet = DiffusionUNet.from_config(config)
        >>> noise_pred = unet(z_noisy, timesteps=t, context=condition_context)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: list[int],
        attention_levels: list[bool],
        num_head_channels: list[int],
        num_res_blocks: int = 2,
        with_conditioning: bool = True,
        cross_attention_dim: int = 512,
        norm_num_groups: int = 32,
    ) -> None:
        super().__init__()

        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            attention_levels=attention_levels,
            num_head_channels=num_head_channels,
            num_res_blocks=num_res_blocks,
            with_conditioning=with_conditioning,
            cross_attention_dim=cross_attention_dim,
            norm_num_groups=norm_num_groups,
        )

    @classmethod
    def from_config(cls, config: dict) -> "DiffusionUNet":
        """Create a DiffusionUNet from a configuration dictionary.

        Args:
            config: Dictionary containing model configuration parameters

        Returns:
            Initialized DiffusionUNet instance
        """
        return cls(
            spatial_dims=config["spatial_dims"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            channels=config["channels"],
            attention_levels=config["attention_levels"],
            num_head_channels=config["num_head_channels"],
            num_res_blocks=config.get("num_res_blocks", 2),
            with_conditioning=config.get("with_conditioning", True),
            cross_attention_dim=config.get("cross_attention_dim", 512),
            norm_num_groups=config.get("norm_num_groups", 32),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass through the diffusion UNet.

        Args:
            x: Noisy latent tensor
            timesteps: Diffusion timesteps
            context: Optional conditioning context

        Returns:
            Predicted noise tensor
        """
        return self.unet(x, timesteps=timesteps, context=context)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> None:
        """Load state dict into the unet."""
        self.unet.load_state_dict(state_dict, strict=strict)

    def state_dict(self) -> dict:
        """Get the state dict of the unet."""
        return self.unet.state_dict()
