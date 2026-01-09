from __future__ import annotations

import random
from typing import Any

import torch
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate

from pti_ldm_vae_v2.vae_regression_common import (
    DEFAULT_NUM_WORKERS,
    build_preprocess_transform,
    filter_attributes_for_paths,
    list_tif_paths,
    select_attribute_sources,
)


class DatasetWithTargets:
    """Wrap a base dataset to attach vector targets."""

    def __init__(self, base_dataset: Dataset, targets: list[torch.Tensor]) -> None:
        """Initialize the dataset wrapper.

        Args:
            base_dataset (Dataset): Underlying dataset returning images.
            targets (list[torch.Tensor]): Target vectors aligned with the dataset order.
        """
        self.base_dataset = base_dataset
        self.targets = [target.clone() for target in targets]

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
            int: Dataset length.
        """
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image tensor and its target vector.

        Args:
            index (int): Sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Image tensor and target vector.
        """
        image = self.base_dataset[index]
        return image, self.targets[index]

    def stacked_targets(self) -> torch.Tensor:
        """Return all targets stacked into a single tensor.

        Returns:
            torch.Tensor: Stacked targets.
        """
        return torch.stack(self.targets, dim=0)

def _attributes_to_tensor(attributes: list[dict[str, float]], targets: list[str]) -> list[torch.Tensor]:
    """Convert attribute dicts to ordered target vectors.

    Args:
        attributes (list[dict[str, float]]): Attributes per image keyed by target name.
        targets (list[str]): Target names defining vector order.

    Returns:
        list[torch.Tensor]: Float tensors with shape [len(targets)].
    """
    vectors: list[torch.Tensor] = []
    for attribute_dict in attributes:
        values = [float(attribute_dict[target]) for target in targets]
        vectors.append(torch.tensor(values, dtype=torch.float32))
    return vectors


def create_regression_dataloaders(
    *,
    data_base_dir: str,
    attributes_path: str | dict[str, str],
    targets: list[str],
    batch_size: int,
    patch_size: tuple[int, int],
    train_split: float = 0.9,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int | None = 42,
    subset_size: int | None = None,
    val_dir: str | None = None,
    cache_rate: float = 0.0,
    data_source: str = "edente",
    normalize_attributes: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """Create train/validation loaders for the regression head.

    Args:
        data_base_dir (str): Root directory with ``edente``/``dente`` folders.
        attributes_path (str | dict[str, str]): JSON path (or mapping per source) with metric values.
        targets (list[str]): Ordered list of targets to predict.
        batch_size (int): Batch size.
        patch_size (tuple[int, int]): Spatial resize target.
        train_split (float): Train/val split ratio when ``val_dir`` is not provided.
        num_workers (int): Number of workers.
        seed (int | None): Seed for shuffling; ``None`` disables shuffling.
        subset_size (int | None): Optional subset for debugging.
        val_dir (str | None): Optional separate validation directory.
        cache_rate (float): Fraction of training set to cache.
        data_source (str): ``edente``, ``dente``, or ``both``.
        normalize_attributes (dict[str, Any] | None): Optional divisor-based normalization.

    Returns:
        tuple[DataLoader, DataLoader, list[str], list[str]]: Train loader, val loader, and file lists.
    """
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(f"cache_rate must be in [0, 1], got {cache_rate}")
    if len(targets) == 0:
        raise ValueError("targets must contain at least one entry.")

    tif_paths = list_tif_paths(data_base_dir, data_source)
    if subset_size is not None:
        tif_paths = tif_paths[:subset_size]

    attribute_sources = select_attribute_sources(attributes_path, data_source)
    attributes_per_image = filter_attributes_for_paths(
        paths=tif_paths,
        attribute_sources=attribute_sources,
        attribute_keys=targets,
        normalize_cfg=normalize_attributes,
    )

    if seed is not None:
        random.seed(seed)
        paired = list(zip(tif_paths, attributes_per_image, strict=True))
        random.shuffle(paired)
        tif_paths, attributes_per_image = zip(*paired, strict=False)
        tif_paths = list(tif_paths)
        attributes_per_image = list(attributes_per_image)

    if val_dir is not None:
        val_paths = list_tif_paths(val_dir, data_source)
        train_paths = tif_paths
        val_attributes = filter_attributes_for_paths(
            paths=list(val_paths),
            attribute_sources=attribute_sources,
            attribute_keys=targets,
            normalize_cfg=normalize_attributes,
        )
        train_attributes = attributes_per_image
    else:
        split_idx = int(train_split * len(tif_paths))
        train_paths = tif_paths[:split_idx]
        val_paths = tif_paths[split_idx:]
        train_attributes = attributes_per_image[:split_idx]
        val_attributes = attributes_per_image[split_idx:]

    train_targets = _attributes_to_tensor(train_attributes, targets)
    val_targets = _attributes_to_tensor(val_attributes, targets)

    transforms = build_preprocess_transform(patch_size)
    if cache_rate > 0:
        train_base = CacheDataset(
            data=train_paths, transform=transforms, cache_rate=cache_rate, num_workers=num_workers
        )
        val_base = CacheDataset(data=val_paths, transform=transforms, cache_rate=1.0, num_workers=num_workers)
    else:
        train_base = Dataset(data=train_paths, transform=transforms)
        val_base = Dataset(data=val_paths, transform=transforms)

    train_ds = DatasetWithTargets(train_base, train_targets)
    val_ds = DatasetWithTargets(val_base, val_targets)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )

    return train_loader, val_loader, train_paths, val_paths


def create_regression_eval_dataloader(
    input_dir: str,
    attributes_path: str | dict[str, str],
    targets: list[str],
    patch_size: tuple[int, int],
    batch_size: int,
    num_workers: int = DEFAULT_NUM_WORKERS,
    num_samples: int | None = None,
    data_source: str = "edente",
    normalize_attributes: dict[str, Any] | None = None,
) -> tuple[DataLoader, list[str]]:
    """Build a dataloader for regression evaluation (images + targets).

    Args:
        input_dir (str): Directory containing images grouped by source.
        attributes_path (str | dict[str, str]): JSON path or mapping of source -> JSON path.
        targets (list[str]): Target names and output ordering.
        patch_size (tuple[int, int]): Spatial resize target.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        num_samples (int | None): Optional cap on number of images.
        data_source (str): Data source key (``edente``, ``dente``, ``both``).
        normalize_attributes (dict[str, Any] | None): Optional divisor-based normalization.

    Returns:
        tuple[DataLoader, list[str]]: Dataloader and image path list.
    """
    tif_paths = list_tif_paths(input_dir, data_source)
    if num_samples is not None:
        tif_paths = tif_paths[:num_samples]

    attribute_sources = select_attribute_sources(attributes_path, data_source)
    attributes_per_image = filter_attributes_for_paths(
        paths=tif_paths,
        attribute_sources=attribute_sources,
        attribute_keys=targets,
        normalize_cfg=normalize_attributes,
    )

    target_tensors = _attributes_to_tensor(attributes_per_image, targets)
    transforms = build_preprocess_transform(patch_size)
    base_ds = Dataset(data=tif_paths, transform=transforms)
    dataset = DatasetWithTargets(base_ds, target_tensors)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    return dataloader, tif_paths


def create_regression_inference_dataloader(
    input_dir: str,
    patch_size: tuple[int, int],
    batch_size: int,
    num_samples: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[DataLoader, list[str]]:
    """Dataloader for regression inference without targets.

    Args:
        input_dir (str): Input directory with images.
        patch_size (tuple[int, int]): Spatial resize target.
        batch_size (int): Batch size.
        num_samples (int | None): Optional cap on number of images.
        num_workers (int): Number of workers.

    Returns:
        tuple[DataLoader, list[str]]: Dataloader and image path list.
    """
    tif_paths = list_tif_paths(input_dir, data_source="both")
    if num_samples is not None:
        tif_paths = tif_paths[:num_samples]

    transforms = build_preprocess_transform(patch_size)
    dataset = Dataset(data=tif_paths, transform=transforms)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=list_data_collate,
    )
    return dataloader, tif_paths
