from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_regression_checkpoint(path: Path, model: nn.Module, targets: list[str], epoch: int | None = None) -> None:
    """Save regression head weights.

    Args:
        path (Path): Destination file.
        model (nn.Module): Model containing ``regressor``.
        targets (list[str]): Ordered target names.
        epoch (int | None): Epoch index for bookkeeping.
    """
    state = {
        "regressor_state_dict": model.regressor.state_dict(),
        "targets": targets,
        "epoch": epoch,
        "latent_dim": getattr(model, "latent_dim", None),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_regression_checkpoint(path: Path, model: nn.Module, expected_targets: list[str]) -> dict[str, Any]:
    """Load regression head weights and validate target order.

    Args:
        path (Path): Checkpoint file.
        model (nn.Module): Model instance to populate.
        expected_targets (list[str]): Target list expected by the caller.

    Returns:
        dict[str, Any]: Metadata stored in the checkpoint (e.g., epoch).

    Raises:
        ValueError: If target ordering mismatches.
    """
    checkpoint = torch.load(path, map_location="cpu")
    stored_targets = checkpoint.get("targets")
    if stored_targets and list(stored_targets) != list(expected_targets):
        raise ValueError(f"Target mismatch: expected {expected_targets}, found {stored_targets}.")

    model.regressor.load_state_dict(checkpoint["regressor_state_dict"])
    return checkpoint


def save_last_regression_checkpoint(weights_dir: Path, model: nn.Module, targets: list[str], epoch: int) -> Path:
    """Persist the latest regression head checkpoint (overwrites).

    Args:
        weights_dir (Path): Directory where weights are stored.
        model (nn.Module): Model containing the regression head.
        targets (list[str]): Ordered target names.
        epoch (int): Current epoch index.

    Returns:
        Path: Path to the saved checkpoint.
    """
    path = weights_dir / "head_last.pth"
    save_regression_checkpoint(path, model, targets, epoch)
    return path


def maybe_save_best_regression_checkpoint(
    weights_dir: Path,
    model: nn.Module,
    targets: list[str],
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    best_path: Path | None = None,
) -> tuple[float, Path]:
    """Save best regression checkpoint if validation improves.

    Args:
        weights_dir (Path): Directory where weights are stored.
        model (nn.Module): Model containing the regression head.
        targets (list[str]): Ordered target names.
        epoch (int): Current epoch index.
        val_loss (float): Current validation loss.
        best_val_loss (float): Best validation loss so far.
        best_path (Path | None): Existing best checkpoint path.

    Returns:
        tuple[float, Path]: Updated best validation loss and the best checkpoint path.
    """
    path = best_path or weights_dir / "head_best.pth"
    if val_loss < best_val_loss:
        save_regression_checkpoint(path, model, targets, epoch)
        return val_loss, path
    return best_val_loss, path
