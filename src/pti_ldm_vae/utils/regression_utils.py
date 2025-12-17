import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pti_ldm_vae.models import LatentRegressor, VAELatentRegressor
from pti_ldm_vae.utils.metrics import compute_regression_metrics
from pti_ldm_vae.utils.vae_loader import load_vae_config, load_vae_model


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
    data_cfg.setdefault("num_workers", config.get("num_workers", legacy_train_cfg.get("num_workers", 4)))
    data_cfg.setdefault("seed", config.get("seed", legacy_train_cfg.get("seed")))
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


def regression_loss_key(train_cfg: dict[str, Any]) -> str:
    """Return a normalized loss key for logging."""
    loss_name = str(train_cfg.get("loss", "mse")).lower()
    if loss_name in {"smooth_l1", "huber"}:
        return "loss_huber"
    return "loss_mse"


def init_regression_wandb(
    config: dict[str, Any],
    run_dir: Path,
    latent_dim: int,
    targets: list[str],
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
) -> Any:
    """Initialize a Weights & Biases run for regression training.

    Args:
        config (dict[str, Any]): Full regression configuration.
        run_dir (Path): Output run directory.
        latent_dim (int): Flattened latent dimension.
        targets (list[str]): Target names.
        data_cfg (dict[str, Any]): Data configuration block.
        train_cfg (dict[str, Any]): Training configuration block.

    Returns:
        Any: ``wandb.run`` instance when enabled, otherwise ``None``.
    """
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    try:
        import wandb  # type: ignore
    except ImportError:
        print("[WARN] W&B is enabled but package 'wandb' is not installed.")
        return None

    project = os.getenv("WANDB_PROJECT", wandb_cfg.get("project", "pti-ldm-vae"))
    entity = wandb_cfg.get("entity") or os.getenv("WANDB_ENTITY")
    run_name = wandb_cfg.get("name") or run_dir.name
    tags = wandb_cfg.get("tags", [])
    notes = wandb_cfg.get("notes", "")

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        tags=tags,
        notes=notes,
        dir=str(run_dir),
        config={
            "architecture": "vae-latent-regressor",
            "latent_dim": latent_dim,
            "targets": targets,
            "hidden_dims": config["regressor_def"].get("hidden_dims", []),
            "dropout": config["regressor_def"].get("dropout", 0.0),
            "activation": config["regressor_def"].get("activation", "relu"),
            "data_source": data_cfg.get("data_source", "edente"),
            "patch_size": data_cfg.get("patch_size"),
            "batch_size": train_cfg.get("batch_size"),
            "lr": train_cfg.get("lr"),
            "max_epochs": train_cfg.get("max_epochs"),
            "target_norm": train_cfg.get("target_norm", "none"),
            "loss": train_cfg.get("loss", "mse"),
        },
    )
    try:
        run.config.update({"full_config_json": config}, allow_val_change=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Could not attach full config to W&B: {exc}")

    print(f"✨ W&B run initialized: {run.url}")
    return run


def log_regression_epoch(
    wandb_run: Any,
    *,
    epoch: int,
    train_loss: float,
    best_val: float,
    val_loss: float | None = None,
    metrics: dict[str, float] | None = None,
    loss_key: str = "loss_mse",
) -> None:
    """Log regression metrics to W&B if enabled.

    Args:
        wandb_run (Any): Active wandb run or ``None``.
        epoch (int): Current epoch index.
        train_loss (float): Training loss.
        best_val (float): Best validation loss so far.
        val_loss (float | None): Validation loss if computed this epoch.
        metrics (dict[str, float] | None): Additional metrics (MAE/MSE).
        loss_key (str): Loss key used for logging (e.g., ``loss_mse`` or ``loss_huber``).
    """
    if wandb_run is None:
        return

    payload: dict[str, Any] = {
        "epoch": epoch,
        f"train/{loss_key}": train_loss,
        f"best/val_{loss_key}": best_val,
    }
    if val_loss is not None:
        payload[f"val/{loss_key}"] = val_loss
    if metrics:
        payload.update({f"val/{k}": v for k, v in metrics.items()})
    try:
        wandb_run.log(payload)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to log to W&B: {exc}")


class TargetNormalizer:
    """Utility to normalize and denormalize target vectors."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """Initialize the normalizer.

        Args:
            mean (torch.Tensor): Mean per target.
            std (torch.Tensor): Standard deviation per target.
        """
        if mean.shape != std.shape:
            raise ValueError("Mean and std must share the same shape.")
        safe_std = torch.where(std == 0, torch.ones_like(std), std)
        self.mean = mean
        self.std = safe_std

    def normalize(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize targets."""
        mean = self.mean.to(targets.device)
        std = self.std.to(targets.device)
        return (targets - mean) / std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Restore normalized values to the original scale."""
        mean = self.mean.to(values.device)
        std = self.std.to(values.device)
        return values * std + mean

    def to_dict(self, target_names: list[str]) -> dict[str, list[float] | list[str]]:
        """Serialize normalizer statistics."""
        return {
            "target_names": target_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[float] | list[str]], target_names: list[str]) -> "TargetNormalizer":
        """Load a normalizer from a dictionary."""
        stored_names = data.get("target_names", [])
        if stored_names and list(stored_names) != target_names:
            raise ValueError(f"Target order mismatch: expected {target_names}, found {stored_names}")

        mean = torch.tensor(data["mean"], dtype=torch.float32)
        std = torch.tensor(data["std"], dtype=torch.float32)
        return cls(mean=mean, std=std)


def compute_target_normalizer(targets: torch.Tensor) -> TargetNormalizer:
    """Compute mean and std for target normalization (standard scaling)."""
    mean = targets.mean(dim=0)
    std = targets.std(dim=0, unbiased=False)
    return TargetNormalizer(mean=mean, std=std)


def save_target_normalizer(path: Path, normalizer: TargetNormalizer, target_names: list[str]) -> None:
    """Persist normalization statistics to JSON."""
    payload = normalizer.to_dict(target_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_target_normalizer(path: Path, target_names: list[str]) -> TargetNormalizer:
    """Load normalization statistics from JSON."""
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return TargetNormalizer.from_dict(data, target_names)


def build_loss_fn(loss_name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Create the regression loss function."""
    if loss_name.lower() in {"mse", "mse_loss"}:
        return nn.MSELoss()
    if loss_name.lower() in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss '{loss_name}'. Use 'mse' or 'smooth_l1'.")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    normalizer: TargetNormalizer | None,
) -> float:
    """Run a single training epoch."""
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
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    target_names: list[str],
    normalizer: TargetNormalizer | None,
) -> tuple[float, dict[str, float]]:
    """Validate model and compute metrics."""
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
    metrics = compute_regression_metrics(stacked_preds, stacked_targets, target_names)
    return total_loss / num_batches, metrics


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


def build_regression_model_from_config(
    config: dict[str, Any], targets: list[str], device: torch.device
) -> tuple[VAELatentRegressor, int]:
    """Instantiate VAE encoder (frozen) and regression head from a config dict.

    Args:
        config (dict[str, Any]): Regression configuration with keys ``vae``, ``regressor_def``/``regressor``, and ``data``/``patch_size``.
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
    in_features = latent_dim

    regressor = LatentRegressor(
        in_features=in_features,
        hidden_dims=reg_cfg.get("hidden_dims", []),
        output_dim=len(targets),
        dropout=float(reg_cfg.get("dropout", 0.0)),
        activation=reg_cfg.get("activation", "relu"),
    )
    model = VAELatentRegressor(
        vae=vae,
        regressor=regressor,
        latent_dim=latent_dim,
    ).to(device)
    return model, latent_dim
