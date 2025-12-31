from typing import Any

import torch


def compute_regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, target_names: list[str]
) -> dict[str, Any]:
    """Compute MAE, MSE, and R2 per target and aggregated.

    Args:
        predictions (torch.Tensor): Model outputs [B, T].
        targets (torch.Tensor): Ground-truth targets [B, T].
        target_names (list[str]): Ordered list of target names.

    Returns:
        dict[str, Any]: Metrics keyed by ``mae``, ``mse``, ``r2`` and per-target entries.
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch between predictions {predictions.shape} and targets {targets.shape}.")

    abs_err = torch.abs(predictions - targets)
    sq_err = (predictions - targets) ** 2

    mae_per_target = torch.mean(abs_err, dim=0)
    mse_per_target = torch.mean(sq_err, dim=0)
    ss_res = torch.sum(sq_err, dim=0)
    target_mean = torch.mean(targets, dim=0)
    ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
    safe_denominator = torch.where(ss_tot > 1e-8, ss_tot, torch.ones_like(ss_tot))
    r2_per_target = torch.where(ss_tot > 1e-8, 1.0 - ss_res / safe_denominator, torch.zeros_like(ss_tot))

    metrics: dict[str, Any] = {
        "mae": float(torch.mean(mae_per_target).item()),
        "mse": float(torch.mean(mse_per_target).item()),
        "r2": float(torch.mean(r2_per_target).item()),
    }

    for idx, name in enumerate(target_names):
        metrics[f"mae_{name}"] = float(mae_per_target[idx].item())
        metrics[f"mse_{name}"] = float(mse_per_target[idx].item())
        metrics[f"r2_{name}"] = float(r2_per_target[idx].item())

    return metrics
