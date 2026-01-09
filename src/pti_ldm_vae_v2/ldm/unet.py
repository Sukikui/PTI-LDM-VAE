from __future__ import annotations

from typing import Any

import torch
from monai.networks.nets import DiffusionModelUNet


class DiffusionUNet(torch.nn.Module):
    """Diffusion UNet wrapper around MONAI's DiffusionModelUNet.

    Args:
        spatial_dims (int): Number of spatial dimensions (2 for 2D, 3 for 3D).
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (list[int]): Channels per resolution level.
        attention_levels (list[bool]): Which levels use attention.
        num_head_channels (list[int]): Number of channels per attention head.
        num_res_blocks (int): Residual blocks per level.
        with_conditioning (bool): Enable cross-attention conditioning.
        cross_attention_dim (int): Cross-attention token dimension.
        norm_num_groups (int): Group norm group count.
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
    def from_config(cls, config: dict[str, Any]) -> DiffusionUNet:
        """Create a DiffusionUNet from a configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            DiffusionUNet: Initialized UNet instance.
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
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the diffusion UNet.

        Args:
            x (torch.Tensor): Noisy latent tensor.
            timesteps (torch.Tensor): Diffusion timesteps.
            context (torch.Tensor | None): Conditioning context.

        Returns:
            torch.Tensor: Predicted noise tensor.
        """
        return self.unet(x, timesteps=timesteps, context=context)

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load state dict into the wrapped UNet.

        Args:
            state_dict (dict[str, Any]): State dictionary to load.
            strict (bool): Whether to enforce strict key matching.
        """
        self.unet.load_state_dict(state_dict, strict=strict)

    def state_dict(self) -> dict[str, Any]:
        """Get the UNet state dictionary.

        Returns:
            dict[str, Any]: State dictionary.
        """
        return self.unet.state_dict()
