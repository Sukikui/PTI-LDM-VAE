from __future__ import annotations

import torch
from torch import nn


def build_loss_fn(loss_name: str) -> nn.Module:
    """Create the regression loss function.

    Args:
        loss_name (str): Loss identifier (``mse`` or ``smooth_l1``).

    Returns:
        nn.Module: Instantiated loss module.
    """
    key = loss_name.lower()
    if key in {"mse", "mse_loss"}:
        return nn.MSELoss()
    if key in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss '{loss_name}'. Use 'mse' or 'smooth_l1'.")


def regression_loss_key(train_cfg: dict[str, object]) -> str:
    """Return a normalized loss key for logging.

    Args:
        train_cfg (dict[str, object]): Training configuration.

    Returns:
        str: Metric key used in logs.
    """
    loss_name = str(train_cfg.get("loss", "mse")).lower()
    if loss_name in {"smooth_l1", "huber"}:
        return "loss_huber"
    return "loss_mse"
