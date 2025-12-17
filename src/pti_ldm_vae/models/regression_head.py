import warnings
from collections.abc import Iterable, Sequence

import torch
from torch import nn

from pti_ldm_vae.models.autoencoder import VAEModel


def _activation_from_name(name: str) -> nn.Module:
    """Return an activation module from a lowercase name.

    Args:
        name (str): Activation name (relu, gelu, leaky_relu, elu).

    Returns:
        nn.Module: Instantiated activation module.
    """
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported activation: {name}. Choose from {', '.join(mapping)}.")
    return mapping[name]()


class LatentRegressor(nn.Module):
    """Configurable MLP that maps flattened latents to target metrics."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        """Initialize the regression head.

        Args:
            in_features (int): Size of the flattened latent vector.
            hidden_dims (Sequence[int]): Hidden layer sizes.
            output_dim (int): Number of regression targets.
            dropout (float): Dropout probability between layers.
            activation (str): Activation name (relu, gelu, leaky_relu, elu).
        """
        super().__init__()
        if in_features <= 0:
            raise ValueError("in_features must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        layers: list[nn.Module] = []
        dims = [in_features, *hidden_dims, output_dim]
        act = _activation_from_name(activation)

        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(act.__class__())  # new instance to avoid shared state
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent_flat: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            latent_flat (torch.Tensor): Flattened latent tensor of shape [B, F].

        Returns:
            torch.Tensor: Regression outputs of shape [B, output_dim].
        """
        return self.mlp(latent_flat)


class VAELatentRegressor(nn.Module):
    """Wrapper combining a frozen VAE encoder and a regression head."""

    def __init__(
        self,
        vae: VAEModel,
        regressor: LatentRegressor,
        *,
        latent_dim: int,
        flatten_warning_threshold: int = 131072,
    ) -> None:
        """Initialize the wrapped model.

        Args:
            vae (VAEModel): Loaded VAE (weights are frozen automatically).
            regressor (LatentRegressor): Regression head.
            latent_dim (int): Flattened latent dimensionality.
            flatten_warning_threshold (int): Warn if flattened latent exceeds this size.
        """
        super().__init__()
        self.vae = vae
        self.regressor = regressor
        self.latent_dim = latent_dim

        expected_input = latent_dim
        first_linear = next((layer for layer in self.regressor.mlp if isinstance(layer, nn.Linear)), None)
        if first_linear is None or first_linear.in_features != expected_input:
            raise ValueError(
                f"Regression head expects in_features={expected_input}, "
                f"got {first_linear.in_features if first_linear else 'unknown'}."
            )

        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()

        self.flatten_warning_threshold = flatten_warning_threshold

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images, flatten latents, and run the regression head.

        Args:
            images (torch.Tensor): Input batch [B, C, H, W].

        Returns:
            torch.Tensor: Predicted metrics [B, num_targets].
        """
        with torch.no_grad():
            latent = self.vae.encode_deterministic(images)

        latent_flat = torch.flatten(latent, start_dim=1)
        if latent_flat.shape[1] > self.flatten_warning_threshold:
            warnings.warn(
                f"Flattened latent dimension {latent_flat.shape[1]} is large; consider reducing patch size or latent channels.",
                stacklevel=2,
            )

        return self.regressor(latent_flat)

    @staticmethod
    def compute_flat_dim(latent: torch.Tensor) -> int:
        """Compute flattened latent size from a latent tensor."""
        return int(torch.flatten(latent, start_dim=1).shape[1])

    @staticmethod
    def infer_flat_dim_from_patch(
        vae: VAEModel,
        patch_size: Iterable[int],
        device: torch.device,
        *,
        channels: int | None = None,
    ) -> int:
        """Run a dummy forward to infer latent flatten size for a given patch.

        Args:
            vae (VAEModel): Frozen VAE model.
            patch_size (Iterable[int]): Spatial size (H, W).
            device (torch.device): Target device for the dummy run.
            channels (int | None): Number of input channels. If ``None``, infer from the VAE.

        Returns:
            int: Flattened latent dimension.
        """
        height, width = patch_size
        inferred_channels = channels if channels is not None else getattr(vae.autoencoder, "in_channels", 1)
        with torch.no_grad():
            dummy = torch.zeros(1, inferred_channels, height, width, device=device)
            latent = vae.encode_deterministic(dummy)
        return VAELatentRegressor.compute_flat_dim(latent)
