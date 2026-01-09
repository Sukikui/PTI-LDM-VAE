from __future__ import annotations

import random
from typing import Any

import torch
from monai.data import DataLoader, Dataset, list_data_collate

from pti_ldm_vae_v2.vae_regression_common import (
    DEFAULT_NUM_WORKERS,
    build_preprocess_transform,
    filter_attributes_for_paths,
    list_tif_paths,
    select_attribute_sources,
)


class DatasetWithAttributes:
    """Attach per-sample attributes to a base MONAI dataset."""

    def __init__(self, base_dataset: Dataset, attributes: list[dict[str, float]]) -> None:
        """Initialize the dataset wrapper.

        Args:
            base_dataset (Dataset): Base dataset returning image tensors.
            attributes (list[dict[str, float]]): Attribute dictionaries aligned with dataset order.
        """
        self.base_dataset = base_dataset
        self.attributes = attributes

    def __len__(self) -> int:
        """Return dataset size.

        Returns:
            int: Dataset length.
        """
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, float]]:
        """Return image tensor and its attribute dictionary.

        Args:
            index (int): Sample index.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: Image tensor and attributes.
        """
        image = self.base_dataset[index]
        return image, self.attributes[index]


def collate_with_attributes(
    batch: list[tuple[torch.Tensor, dict[str, float]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Collate a batch of images and attribute dictionaries.

    Args:
        batch (list[tuple[torch.Tensor, dict[str, float]]]): Batch of (image, attrs).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: Stacked images and per-attribute tensors.
    """
    images = torch.stack([item[0] for item in batch], dim=0)
    attribute_keys = batch[0][1].keys()
    attributes = {
        key: torch.tensor([float(item[1][key]) for item in batch], dtype=torch.float32) for key in attribute_keys
    }
    return images, attributes


def create_inference_dataloader(
    input_dir: str,
    patch_size: tuple[int, int],
    batch_size: int,
    num_samples: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[DataLoader, list[str]]:
    """Create a dataloader for VAE inference or evaluation.

    Args:
        input_dir (str): Directory containing input images.
        patch_size (tuple[int, int]): Resize target.
        batch_size (int): Batch size.
        num_samples (int | None): Optional cap on number of images.
        num_workers (int): Number of dataloader workers.

    Returns:
        tuple[DataLoader, list[str]]: Dataloader and list of image paths.
    """
    tif_paths = list_tif_paths(input_dir, data_source="both")
    if num_samples is not None:
        tif_paths = tif_paths[:num_samples]

    transforms = build_preprocess_transform(patch_size)
    dataset = Dataset(data=tif_paths, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader, tif_paths


def create_train_val_dataloaders(
    *,
    data_base_dir: str,
    batch_size: int,
    patch_size: tuple[int, int],
    data_source: str = "edente",
    train_split: float = 0.9,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int | None = 42,
    subset_size: int | None = None,
    val_dir: str | None = None,
    ar_vae_enabled: bool = False,
    regularized_attributes: dict[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """Create train/val dataloaders for VAE training.

    Args:
        data_base_dir (str): Root directory containing image subfolders.
        batch_size (int): Batch size.
        patch_size (tuple[int, int]): Target spatial size.
        data_source (str): ``edente``, ``dente``, or ``both``.
        train_split (float): Train/val split ratio.
        num_workers (int): Number of dataloader workers.
        seed (int | None): Seed for shuffling.
        subset_size (int | None): Optional subset for debugging.
        val_dir (str | None): Optional separate validation directory.
        ar_vae_enabled (bool): Whether AR-VAE is enabled.
        regularized_attributes (dict[str, Any] | None): AR-VAE configuration block.

    Returns:
        tuple[DataLoader, DataLoader, list[str], list[str]]: Train loader, val loader, train paths, val paths.
    """
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")

    tif_paths = list_tif_paths(data_base_dir, data_source)
    if subset_size is not None:
        tif_paths = tif_paths[:subset_size]

    attributes_per_image: list[dict[str, float]] | None = None
    train_attributes: list[dict[str, float]] | None = None
    val_attributes: list[dict[str, float]] | None = None
    attribute_sources: dict[str, dict[str, dict[str, float]]] | None = None
    attribute_keys: list[str] | None = None
    normalize_cfg: dict[str, Any] | None = None

    if ar_vae_enabled:
        if regularized_attributes is None:
            raise ValueError("AR-VAE enabled but regularized_attributes config is missing.")

        attribute_file_cfg = regularized_attributes.get("attribute_file")
        raw_mapping = regularized_attributes.get("attribute_latent_mapping", {})
        attribute_keys = [key for key in raw_mapping if not str(key).startswith("_")]
        if not attribute_keys:
            raise ValueError("attribute_latent_mapping must be provided when AR-VAE is enabled.")

        attribute_sources = select_attribute_sources(attribute_file_cfg, data_source)
        normalize_cfg = regularized_attributes.get("normalize_attributes")

        attributes_per_image = filter_attributes_for_paths(
            paths=tif_paths,
            attribute_sources=attribute_sources,
            attribute_keys=attribute_keys,
            normalize_cfg=normalize_cfg,
        )

    if seed is not None:
        random.seed(seed)
        if attributes_per_image is not None:
            paired = list(zip(tif_paths, attributes_per_image, strict=True))
            random.shuffle(paired)
            tif_paths, attributes_per_image = zip(*paired, strict=False)
            tif_paths = list(tif_paths)
            attributes_per_image = list(attributes_per_image)
        else:
            random.shuffle(tif_paths)

    if val_dir is not None:
        val_paths = list_tif_paths(val_dir, data_source)
        train_paths = tif_paths
        if attributes_per_image is not None:
            train_attributes = attributes_per_image
            if attribute_sources is None or attribute_keys is None:
                raise ValueError("Attribute sources missing for AR-VAE validation.")
            val_attributes = filter_attributes_for_paths(
                paths=list(val_paths),
                attribute_sources=attribute_sources,
                attribute_keys=attribute_keys,
                normalize_cfg=normalize_cfg,
            )
    else:
        split_idx = int(train_split * len(tif_paths))
        train_paths = tif_paths[:split_idx]
        val_paths = tif_paths[split_idx:]
        if attributes_per_image is not None:
            train_attributes = attributes_per_image[:split_idx]
            val_attributes = attributes_per_image[split_idx:]

    transforms = build_preprocess_transform(patch_size)
    train_base = Dataset(data=train_paths, transform=transforms)
    val_base = Dataset(data=val_paths, transform=transforms)

    if ar_vae_enabled:
        if train_attributes is None or val_attributes is None:
            raise ValueError("Attributes must be available when AR-VAE is enabled.")
        train_ds = DatasetWithAttributes(train_base, list(train_attributes))
        val_ds = DatasetWithAttributes(val_base, list(val_attributes))
        collate_fn = collate_with_attributes
    else:
        train_ds = train_base
        val_ds = val_base
        collate_fn = list_data_collate

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_paths, val_paths
