import torch


def compute_kl_loss(
    z_mu: torch.Tensor,
    z_logvar: torch.Tensor,
    *,
    input_is_logvar: bool = True,
) -> torch.Tensor:
    """Compute KL divergence loss for a diagonal Gaussian posterior.

    MONAI AutoencoderKL returns ``(reconstruction, mean, logvar)``; by default this
    function assumes the second argument is a log-variance tensor. If you pass a
    standard deviation tensor instead, set ``input_is_logvar=False``.

    Args:
        z_mu: Mean of the latent distribution [B, C, ...].
        z_logvar: Log-variance (default) or standard deviation tensor.
        input_is_logvar: When ``True`` (default), treats ``z_logvar`` as log-variance.
            When ``False``, treats ``z_logvar`` as sigma and converts to log-variance.

    Returns:
        KL divergence loss (scalar, averaged over batch).
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
    """Compose the total loss with consistent weighting for train/validation.

    Args:
        recons_loss: Reconstruction loss (intensity component).
        kl_loss: KL divergence term.
        perceptual_loss: Perceptual loss term.
        adv_gen_loss: Adversarial generator loss term.
        ar_loss: Attribute-regularization loss term.
        kl_weight: Weight applied to KL term.
        perceptual_weight: Weight applied to perceptual term.
        adv_weight: Weight applied to adversarial generator term.
        ar_gamma: Weight applied to AR term.
        ar_vae_enabled: Whether AR-VAE is enabled.

    Returns:
        Total loss tensor.
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
        latent_vectors: Latent tensor of shape [B, C] or [B, C, H, W].
        attributes: Mapping attribute name -> tensor of shape [B].
        attribute_latent_mapping: Config mapping with latent_channel and optional delta.
        pairwise_mode: "all" or "subset".
        subset_pairs: Number of pairs to sample if pairwise_mode == "subset".
        delta_global: Optional global delta config with keys enabled/value.

    Returns:
        Tuple of (total_loss, losses_per_attr, pair_counts, deltas_per_attr).

    Raises:
        ValueError: On invalid config or latent dimension mismatch.
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
