from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .normalization import TargetNormalizer


def _compute_regression_metrics(
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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    normalizer: TargetNormalizer | None,
) -> float:
    """Run a single training epoch.

    Args:
        model (nn.Module): Regression model.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): Optimizer for trainable params.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Target device.
        normalizer (TargetNormalizer | None): Optional target normalizer.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        targets_for_loss = normalizer.normalize(targets) if normalizer is not None else targets

        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_fn(predictions, targets_for_loss)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Training dataloader produced zero batches.")
    return total_loss / num_batches


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_names: list[str],
    normalizer: TargetNormalizer | None,
) -> tuple[float, dict[str, float]]:
    """Validate model and compute metrics.

    Args:
        model (nn.Module): Regression model.
        dataloader (DataLoader): Validation dataloader.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Target device.
        target_names (list[str]): Ordered target names.
        normalizer (TargetNormalizer | None): Optional target normalizer.

    Returns:
        tuple[float, dict[str, float]]: Average validation loss and metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            targets_for_loss = normalizer.normalize(targets) if normalizer is not None else targets

            predictions = model(images)
            loss = loss_fn(predictions, targets_for_loss)
            total_loss += loss.item()
            num_batches += 1

            if normalizer is not None:
                predictions = normalizer.denormalize(predictions)

            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())

    if num_batches == 0:
        raise RuntimeError("Validation dataloader produced zero batches.")

    stacked_preds = torch.cat(all_preds, dim=0)
    stacked_targets = torch.cat(all_targets, dim=0)
    metrics = _compute_regression_metrics(stacked_preds, stacked_targets, target_names)
    return total_loss / num_batches, metrics
