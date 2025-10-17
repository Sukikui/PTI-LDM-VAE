import torch


def compute_kl_loss(z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence loss for Variational Autoencoder.

    Args:
        z_mu: Mean of the latent distribution [B, C, ...]
        z_sigma: Standard deviation of the latent distribution [B, C, ...]

    Returns:
        KL divergence loss (scalar)
    """
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]
