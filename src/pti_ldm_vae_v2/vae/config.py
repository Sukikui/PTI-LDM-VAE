from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from monai.bundle import ConfigParser

from pti_ldm_vae_v2.common import VAEModel
from pti_ldm_vae_v2.common import resolve_run_subdir


def load_config(config_file: str) -> SimpleNamespace:
    """Load a VAE configuration file with MONAI's ConfigParser.

    Args:
        config_file (str): Path to the configuration JSON file.

    Returns:
        SimpleNamespace: Parsed configuration with resolved references.
    """
    parser = ConfigParser()
    parser.read_config(config_file)
    parser.parse(True)
    cfg_dict = parser.get_parsed_content()
    return SimpleNamespace(**cfg_dict)


def apply_overrides(
    config: SimpleNamespace,
    *,
    batch_size: int | None = None,
    lr: float | None = None,
    max_epochs: int | None = None,
) -> None:
    """Apply CLI overrides to the loaded configuration.

    Args:
        config (SimpleNamespace): Parsed configuration object.
        batch_size (int | None): Optional batch size override.
        lr (float | None): Optional learning rate override.
        max_epochs (int | None): Optional max epochs override.
    """
    train_cfg = config.autoencoder_train
    if batch_size is not None:
        train_cfg["batch_size"] = batch_size
    if lr is not None:
        train_cfg["lr"] = lr
    if max_epochs is not None:
        train_cfg["max_epochs"] = max_epochs


def resolve_run_dirs(run_dir: str | Path) -> tuple[Path, Path, Path]:
    """Create and return VAE run directories.

    Args:
        run_dir (str | Path): Base run directory.

    Returns:
        tuple[Path, Path, Path]: Tuple of (run_dir, weights_dir, splits_dir).
    """
    run_path = Path(run_dir)
    weights_dir = run_path / "trained_weights"
    splits_dir = run_path / "splits"
    run_path.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    return run_path, weights_dir, splits_dir


def load_model(config: Any, checkpoint_path: str, device: torch.device) -> VAEModel:
    """Load a VAE model from checkpoint into evaluation mode.

    Args:
        config (Any): Parsed configuration containing ``autoencoder_def``.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Target device.

    Returns:
        VAEModel: Loaded VAE model in eval mode.
    """
    autoencoder = VAEModel.from_config(config.autoencoder_def).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("autoencoder_state_dict", checkpoint)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    return autoencoder


def load_config_and_model(
    config_file: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[SimpleNamespace, VAEModel]:
    """Load a VAE config and checkpoint in one step.

    Args:
        config_file (str): Path to the VAE config JSON.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Target device.

    Returns:
        tuple[SimpleNamespace, VAEModel]: Parsed config and loaded VAE model.
    """
    config = load_config(config_file)
    autoencoder = load_model(config, checkpoint_path, device)
    return config, autoencoder


def default_eval_output_dir(
    config_file: str | None = None,
    *,
    run_dir: str | Path | None = None,
    input_dir: str | Path | None = None,
    root_dir: str = "evals",
) -> Path:
    """Compute the default output directory for evaluation artifacts.

    Args:
        config_file (str | None): Path to the configuration file (legacy fallback).
        run_dir (str | Path | None): Run directory for the current model.
        input_dir (str | Path | None): Input directory to mirror under ``run_dir``.
        root_dir (str): Root directory used for legacy evaluations.

    Returns:
        Path: Folder path for evaluation artifacts.
    """
    if run_dir is not None and input_dir is not None:
        return resolve_run_subdir(run_dir, input_dir, "eval")
    if config_file is None:
        raise ValueError("config_file is required when run_dir/input_dir are not provided.")
    config_name = Path(config_file).stem
    return Path(root_dir) / config_name
