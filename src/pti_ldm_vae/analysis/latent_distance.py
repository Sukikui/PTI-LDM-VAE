from __future__ import annotations

import numpy as np


def latent_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute Euclidean distance between two latent vectors."""
    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError(f"Expected 1D latent vectors, got shapes {vec_a.shape} and {vec_b.shape}")
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Latent vectors must have the same shape, got {vec_a.shape} and {vec_b.shape}")
    return float(np.linalg.norm(vec_a - vec_b))


def latent_distance_from_indices(latents: np.ndarray, idx_a: int, idx_b: int) -> float:
    """Compute Euclidean distance between two latent vectors from the same group using indices."""
    if latents.ndim != 2:
        raise ValueError(f"Expected latents of shape [N, D], got shape {latents.shape}")
    if not (0 <= idx_a < latents.shape[0] and 0 <= idx_b < latents.shape[0]):
        raise ValueError(f"indices must be in [0, {latents.shape[0] - 1}], got {idx_a} and {idx_b}")
    return latent_distance(latents[idx_a], latents[idx_b])


def latent_distance_cross(
    latents_a: np.ndarray,
    idx_a: int,
    latents_b: np.ndarray,
    idx_b: int,
) -> float:
    """Compute Euclidean distance between two latent vectors from different groups using indices."""
    if latents_a.ndim != 2 or latents_b.ndim != 2:
        raise ValueError(f"Expected 2D latents for both groups, got shapes {latents_a.shape} and {latents_b.shape}")
    if latents_a.shape[1] != latents_b.shape[1]:
        raise ValueError(
            f"Latent dimensions must match between groups, got {latents_a.shape[1]} and {latents_b.shape[1]}"
        )
    if not (0 <= idx_a < latents_a.shape[0]):
        raise ValueError(f"idx_a must be in [0, {latents_a.shape[0] - 1}], got {idx_a}")
    if not (0 <= idx_b < latents_b.shape[0]):
        raise ValueError(f"idx_b must be in [0, {latents_b.shape[0] - 1}], got {idx_b}")
    return latent_distance(latents_a[idx_a], latents_b[idx_b])
