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
