from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_file: str) -> dict[str, Any]:
    """Load an LDM configuration JSON file.

    Args:
        config_file (str): Path to the LDM configuration file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_file, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run_dir(config: dict[str, Any], config_file: str) -> Path:
    """Resolve and create the run directory.

    Args:
        config (dict[str, Any]): LDM configuration dictionary.
        config_file (str): Path to the config file for fallback naming.

    Returns:
        Path: Run directory path.
    """
    if config.get("run_dir"):
        run_dir = Path(config["run_dir"])
    else:
        run_dir = Path("runs") / Path(config_file).stem
        config["run_dir"] = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_run_dirs(run_dir: Path) -> tuple[Path, Path, Path]:
    """Create and return run subdirectories.

    Args:
        run_dir (Path): Base run directory.

    Returns:
        tuple[Path, Path, Path]: (run_dir, weights_dir, splits_dir).
    """
    weights_dir = run_dir / "trained_weights"
    splits_dir = run_dir / "splits"
    weights_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, weights_dir, splits_dir


def apply_train_overrides(
    train_cfg: dict[str, Any],
    *,
    batch_size: int | None,
    lr: float | None,
    max_epochs: int | None,
) -> dict[str, Any]:
    """Apply CLI overrides to the training configuration.

    Args:
        train_cfg (dict[str, Any]): Training configuration block.
        batch_size (int | None): Optional batch size override.
        lr (float | None): Optional learning rate override.
        max_epochs (int | None): Optional max epochs override.

    Returns:
        dict[str, Any]: Updated training configuration.
    """
    updated = dict(train_cfg)
    if batch_size is not None:
        updated["batch_size"] = batch_size
    if lr is not None:
        updated["lr"] = lr
    if max_epochs is not None:
        updated["max_epochs"] = max_epochs
    return updated
