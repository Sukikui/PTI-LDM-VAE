from __future__ import annotations

import json
from pathlib import Path

import torch

NORM_STATS_FILENAME = "target_norm_stats.json"


class TargetNormalizer:
    """Utility to normalize and denormalize target vectors."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
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
        """Normalize targets.

        Args:
            targets (torch.Tensor): Input targets.

        Returns:
            torch.Tensor: Normalized targets.
        """
        mean = self.mean.to(targets.device)
        std = self.std.to(targets.device)
        return (targets - mean) / std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        """Restore normalized values to the original scale.

        Args:
            values (torch.Tensor): Normalized values.

        Returns:
            torch.Tensor: Denormalized values.
        """
        mean = self.mean.to(values.device)
        std = self.std.to(values.device)
        return values * std + mean

    def to_dict(self, target_names: list[str]) -> dict[str, list[float] | list[str]]:
        """Serialize normalizer statistics.

        Args:
            target_names (list[str]): Ordered target names.

        Returns:
            dict[str, list[float] | list[str]]: Serializable payload.
        """
        return {
            "target_names": target_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[float] | list[str]], target_names: list[str]) -> "TargetNormalizer":
        """Load a normalizer from a dictionary.

        Args:
            data (dict[str, list[float] | list[str]]): Serialized normalizer payload.
            target_names (list[str]): Expected target ordering.

        Returns:
            TargetNormalizer: Restored normalizer instance.
        """
        stored_names = data.get("target_names", [])
        if stored_names and list(stored_names) != target_names:
            raise ValueError(f"Target order mismatch: expected {target_names}, found {stored_names}")

        mean = torch.tensor(data["mean"], dtype=torch.float32)
        std = torch.tensor(data["std"], dtype=torch.float32)
        return cls(mean=mean, std=std)


def compute_target_normalizer(targets: torch.Tensor) -> TargetNormalizer:
    """Compute mean and std for target normalization (standard scaling).

    Args:
        targets (torch.Tensor): Target tensor [B, T].

    Returns:
        TargetNormalizer: Normalizer instance.
    """
    mean = targets.mean(dim=0)
    std = targets.std(dim=0, unbiased=False)
    return TargetNormalizer(mean=mean, std=std)


def save_target_normalizer(path: Path, normalizer: TargetNormalizer, target_names: list[str]) -> None:
    """Persist normalization statistics to JSON.

    Args:
        path (Path): Destination file path.
        normalizer (TargetNormalizer): Normalizer to serialize.
        target_names (list[str]): Ordered target names.
    """
    payload = normalizer.to_dict(target_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_target_normalizer(path: Path, target_names: list[str]) -> TargetNormalizer:
    """Load normalization statistics from JSON.

    Args:
        path (Path): Path to the JSON file.
        target_names (list[str]): Expected target ordering.

    Returns:
        TargetNormalizer: Loaded normalizer.
    """
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return TargetNormalizer.from_dict(data, target_names)
