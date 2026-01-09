from __future__ import annotations

import torch


def select_intensity_loss(recon_loss: str | None, *, verbose: bool = False) -> torch.nn.Module:
    """Select the intensity loss module (L1 or L2).

    Args:
        recon_loss (str | None): Loss identifier (``"l2"`` selects MSE; otherwise L1).
        verbose (bool): Whether to print which loss is selected.

    Returns:
        torch.nn.Module: Instantiated loss module.
    """
    if recon_loss == "l2":
        if verbose:
            print("Using L2 loss")
        return torch.nn.MSELoss()
    if verbose:
        print("Using L1 loss")
    return torch.nn.L1Loss()


def ensure_three_channels(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor with three channels by repeating single-channel inputs.

    Args:
        tensor (torch.Tensor): Tensor of shape [B, C, H, W] passed to perceptual loss.

    Returns:
        torch.Tensor: Tensor with three channels suitable for ImageNet backbones.

    Raises:
        ValueError: If ``tensor`` is not four-dimensional or has an unsupported channel count.
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {tensor.shape}")

    channels = tensor.shape[1]
    if channels == 3:
        return tensor
    if channels == 1:
        return tensor.repeat(1, 3, 1, 1)
    raise ValueError(f"Perceptual loss expects 1 or 3 channels, got {channels}")


def compute_kl_loss(
    z_mu: torch.Tensor,
    z_logvar: torch.Tensor,
    *,
    input_is_logvar: bool = True,
) -> torch.Tensor:
    """Compute KL divergence loss for a diagonal Gaussian posterior.

    Args:
        z_mu (torch.Tensor): Mean of the latent distribution [B, C, ...].
        z_logvar (torch.Tensor): Log-variance (default) or standard deviation tensor.
        input_is_logvar (bool): Treat ``z_logvar`` as log-variance when True.

    Returns:
        torch.Tensor: KL divergence loss (scalar, averaged over batch).
    """
    if not input_is_logvar:
        z_logvar = torch.log(z_logvar.pow(2) + 1e-8)

    dim = list(range(1, z_logvar.dim()))
    kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar), dim=dim)
    return kl.mean()


def compute_total_loss(
    recons_loss: torch.Tensor,
    kl_loss: torch.Tensor,
    perceptual_loss: torch.Tensor,
    adv_gen_loss: torch.Tensor,
    ar_loss: torch.Tensor,
    *,
    kl_weight: float,
    perceptual_weight: float,
    adv_weight: float,
    ar_gamma: float,
    ar_vae_enabled: bool,
) -> torch.Tensor:
    """Compose the total loss with consistent weighting.

    Args:
        recons_loss (torch.Tensor): Reconstruction loss (intensity component).
        kl_loss (torch.Tensor): KL divergence term.
        perceptual_loss (torch.Tensor): Perceptual loss term.
        adv_gen_loss (torch.Tensor): Adversarial generator loss term.
        ar_loss (torch.Tensor): Attribute-regularization loss term.
        kl_weight (float): Weight applied to KL term.
        perceptual_weight (float): Weight applied to perceptual term.
        adv_weight (float): Weight applied to adversarial generator term.
        ar_gamma (float): Weight applied to AR term.
        ar_vae_enabled (bool): Whether AR-VAE is enabled.

    Returns:
        torch.Tensor: Total loss tensor.
    """
    total = recons_loss + kl_weight * kl_loss + perceptual_weight * perceptual_loss + adv_weight * adv_gen_loss
    if ar_vae_enabled:
        total = total + ar_gamma * ar_loss
    return total


def compute_ar_vae_loss(
    latent_vectors: torch.Tensor,
    attributes: dict[str, torch.Tensor],
    attribute_latent_mapping: dict[str, dict[str, float]],
    pairwise_mode: str,
    subset_pairs: int | None,
    delta_global: dict[str, float] | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int], dict[str, float]]:
    """Compute Attribute-Regularized VAE loss.

    Args:
        latent_vectors (torch.Tensor): Latent tensor of shape [B, C] or [B, C, H, W].
        attributes (dict[str, torch.Tensor]): Mapping attribute name -> tensor of shape [B].
        attribute_latent_mapping (dict[str, dict[str, float]]): Config mapping with latent_channel and optional delta.
        pairwise_mode (str): "all" or "subset".
        subset_pairs (int | None): Number of pairs to sample if pairwise_mode == "subset".
        delta_global (dict[str, float] | None): Optional global delta config with keys enabled/value.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, int], dict[str, float]]: Total loss, per-attr losses,
        pair counts, and delta values.
    """
    if latent_vectors.dim() == 4:
        latent_vectors = latent_vectors.mean(dim=(2, 3))
    elif latent_vectors.dim() != 2:
        raise ValueError(f"Expected latent shape [B, C] or [B, C, H, W], got {latent_vectors.shape}")

    batch_size, latent_dim = latent_vectors.shape
    if pairwise_mode not in {"all", "subset"}:
        raise ValueError(f"pairwise must be 'all' or 'subset', got {pairwise_mode}")

    if pairwise_mode == "subset":
        if subset_pairs is None or subset_pairs <= 0:
            raise ValueError("subset_pairs must be a positive integer when pairwise='subset'")

    total_loss = torch.tensor(0.0, device=latent_vectors.device)
    losses_per_attr: dict[str, torch.Tensor] = {}
    pair_counts: dict[str, int] = {}
    deltas_per_attr: dict[str, float] = {}

    for attr_name, mapping in attribute_latent_mapping.items():
        target_latent = int(mapping["latent_channel"])
        if target_latent >= latent_dim:
            raise ValueError(
                f"Latent channel {target_latent} for attribute {attr_name} exceeds latent size {latent_dim}"
            )

        attr_values = attributes.get(attr_name)
        if attr_values is None:
            raise KeyError(f"Missing attribute values for {attr_name} in batch.")

        attr_values = attr_values.to(latent_vectors.device)

        delta_attr = mapping.get("delta")
        if delta_attr is None and delta_global and delta_global.get("enabled", False):
            delta_attr = delta_global.get("value")
        if delta_attr is None:
            raise ValueError(f"Delta not provided for {attr_name} and no delta_global fallback.")

        latent_component = latent_vectors[:, target_latent]

        all_pairs = [(i, j) for i in range(batch_size) for j in range(batch_size) if i != j]
        if pairwise_mode == "subset":
            import random

            all_pairs = random.sample(all_pairs, min(len(all_pairs), int(subset_pairs)))

        if len(all_pairs) == 0:
            losses_per_attr[attr_name] = torch.tensor(0.0, device=latent_vectors.device)
            pair_counts[attr_name] = 0
            deltas_per_attr[attr_name] = float(delta_attr)
            continue

        idx_i = torch.tensor([pair[0] for pair in all_pairs], device=latent_vectors.device)
        idx_j = torch.tensor([pair[1] for pair in all_pairs], device=latent_vectors.device)

        delta_a = attr_values[idx_j] - attr_values[idx_i]
        ordering = torch.sign(delta_a)
        mask = ordering != 0

        if not torch.any(mask):
            losses_per_attr[attr_name] = torch.tensor(0.0, device=latent_vectors.device)
            pair_counts[attr_name] = 0
            deltas_per_attr[attr_name] = float(delta_attr)
            continue

        delta_z = latent_component[idx_j] - latent_component[idx_i]
        pred = torch.tanh(float(delta_attr) * delta_z[mask])
        loss_attr = torch.mean((pred - ordering[mask]) ** 2)

        losses_per_attr[attr_name] = loss_attr
        pair_counts[attr_name] = int(mask.sum().item())
        deltas_per_attr[attr_name] = float(delta_attr)
        total_loss = total_loss + loss_attr

    return total_loss, losses_per_attr, pair_counts, deltas_per_attr
