from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from monai.bundle import ConfigParser

from pti_ldm_vae_v2.vae_regression_common import (
    DEFAULT_NUM_WORKERS,
    LatentRegressor,
    VAELatentRegressor,
    VAEModel,
)


def load_config(config_file: str) -> dict[str, Any]:
    """Load a regression head configuration from JSON.

    Args:
        config_file (str): Path to the JSON configuration file.

    Returns:
        dict[str, Any]: Parsed configuration dictionary.
    """
    with open(config_file, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_run_dir(config: dict[str, Any], config_file: str) -> Path:
    """Resolve the run directory and ensure it exists.

    Args:
        config (dict[str, Any]): Configuration dictionary (mutated in-place when missing ``run_dir``).
        config_file (str): Path to the configuration file (used for fallback).

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


def extract_regression_data_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize regression data configuration across schemas.

    Args:
        config (dict[str, Any]): Full regression configuration.

    Returns:
        dict[str, Any]: Data configuration with required keys set.

    Raises:
        KeyError: If mandatory fields are missing.
    """
    data_cfg = dict(config.get("data", {}))
    legacy_train_cfg = config.get("train", {})
    data_cfg.setdefault("data_base_dir", config.get("data_base_dir"))
    data_cfg.setdefault("attributes_path", config.get("attributes_path"))
    data_cfg.setdefault("data_source", config.get("data_source", "edente"))
    data_cfg.setdefault("train_split", config.get("train_split", 0.9))
    data_cfg.setdefault("val_dir", config.get("val_dir"))
    data_cfg.setdefault("patch_size", config.get("patch_size"))
    data_cfg.setdefault("cache_rate", config.get("cache_rate", legacy_train_cfg.get("cache_rate", 0.0)))
    data_cfg.setdefault("num_workers", config.get("num_workers", legacy_train_cfg.get("num_workers", DEFAULT_NUM_WORKERS)))
    data_cfg.setdefault("seed", config.get("seed", legacy_train_cfg.get("seed", 42)))
    data_cfg.setdefault("subset_size", config.get("subset_size", legacy_train_cfg.get("subset_size")))
    data_cfg.setdefault("normalize_attributes", config.get("normalize_attributes"))

    required = ["data_base_dir", "attributes_path", "patch_size"]
    missing = [field for field in required if data_cfg.get(field) is None]
    if missing:
        raise KeyError(f"Missing required data config fields: {missing}")

    return data_cfg


def extract_regression_train_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize regression training configuration across schemas.

    Args:
        config (dict[str, Any]): Full regression configuration.

    Returns:
        dict[str, Any]: Training configuration with defaults applied.

    Raises:
        KeyError: If required training parameters are missing.
    """
    train_cfg = dict(config.get("regression_train") or config.get("train") or {})
    required = ["batch_size", "lr", "max_epochs"]
    missing = [field for field in required if train_cfg.get(field) is None]
    if missing:
        raise KeyError(f"Missing required training config fields: {missing}")

    train_cfg.setdefault("val_interval", 1)
    train_cfg.setdefault("target_norm", "none")
    train_cfg.setdefault("loss", "mse")
    train_cfg.setdefault("weight_decay", 0.0)
    return train_cfg


def extract_regression_eval_config(config: dict[str, Any], data_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Normalize evaluation configuration, falling back to training data settings.

    Args:
        config (dict[str, Any]): Full regression configuration.
        data_cfg (dict[str, Any] | None): Optional precomputed data configuration for defaults.

    Returns:
        dict[str, Any]: Evaluation configuration with required fields set.

    Raises:
        KeyError: If required evaluation parameters are missing.
    """
    base_data_cfg = data_cfg or extract_regression_data_config(config)
    eval_cfg = dict(config.get("evaluation", {}))

    eval_cfg.setdefault("data_base_dir", base_data_cfg.get("data_base_dir"))
    eval_cfg.setdefault("attributes_path", base_data_cfg.get("attributes_path"))
    eval_cfg.setdefault("data_source", base_data_cfg.get("data_source", "edente"))
    eval_cfg.setdefault("patch_size", base_data_cfg.get("patch_size"))
    eval_cfg.setdefault("num_workers", base_data_cfg.get("num_workers", DEFAULT_NUM_WORKERS))
    eval_cfg.setdefault("normalize_attributes", base_data_cfg.get("normalize_attributes"))

    required = ["data_base_dir", "attributes_path", "patch_size"]
    missing = [field for field in required if eval_cfg.get(field) is None]
    if missing:
        raise KeyError(f"Missing required evaluation config fields: {missing}")

    return eval_cfg


def extract_regressor_def_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize regression head definition configuration across schemas.

    Args:
        config (dict[str, Any]): Full regression configuration.

    Returns:
        dict[str, Any]: Regressor configuration with defaults applied.
    """
    reg_cfg = dict(config.get("regressor_def") or config.get("regressor") or {})
    reg_cfg.setdefault("hidden_dims", [])
    reg_cfg.setdefault("dropout", 0.0)
    reg_cfg.setdefault("activation", "relu")
    return reg_cfg


def load_vae_config(config_file: str) -> SimpleNamespace:
    """Load a VAE configuration file with MONAI's ConfigParser.

    Args:
        config_file (str): Path to the VAE configuration JSON file.

    Returns:
        SimpleNamespace: Parsed configuration with resolved references.
    """
    parser = ConfigParser()
    parser.read_config(config_file)
    parser.parse(True)
    cfg_dict = parser.get_parsed_content()
    return SimpleNamespace(**cfg_dict)


def load_vae_model(config: Any, checkpoint_path: str, device: torch.device) -> VAEModel:
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


def build_regression_model_from_config(
    config: dict[str, Any],
    targets: list[str],
    device: torch.device,
) -> tuple[VAELatentRegressor, int]:
    """Instantiate the VAE encoder (frozen) and regression head from config.

    Args:
        config (dict[str, Any]): Regression configuration with VAE and regressor settings.
        targets (list[str]): Target names defining output dimension.
        device (torch.device): Device for the VAE and head.

    Returns:
        tuple[VAELatentRegressor, int]: Wrapped model and flattened latent dimension.
    """
    vae_cfg = load_vae_config(config["vae"]["config_file"])
    vae = load_vae_model(vae_cfg, config["vae"]["checkpoint"], device)

    data_cfg = extract_regression_data_config(config)
    reg_cfg = extract_regressor_def_config(config)
    patch_size = tuple(data_cfg["patch_size"])

    latent_dim = VAELatentRegressor.infer_flat_dim_from_patch(vae, patch_size, device)
    regressor = LatentRegressor(
        in_features=latent_dim,
        hidden_dims=reg_cfg.get("hidden_dims", []),
        output_dim=len(targets),
        dropout=float(reg_cfg.get("dropout", 0.0)),
        activation=reg_cfg.get("activation", "relu"),
    )
    model = VAELatentRegressor(vae=vae, regressor=regressor, latent_dim=latent_dim).to(device)
    return model, latent_dim
