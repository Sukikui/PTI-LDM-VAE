import json
import os
import random
from collections.abc import Callable
from glob import glob
from typing import Any

import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImage,
    LoadImaged,
    Resize,
    ResizeD,
)

from .transforms import ApplyLocalNormd, LocalNormalizeByMask, ToTuple


class IdentityTensorTransform:
    """Return data unchanged (picklable helper for DataLoader workers)."""

    def __call__(self, data: Any) -> Any:
        """Forward data without modifications.

        Args:
            data (Any): Input data element.

        Returns:
            Any: Same object without changes.
        """
        return data


class AlbumentationsSingleChannel:
    """Apply an albumentations transform to single-channel tensors."""

    def __init__(self, albumentations_transform: Callable):
        """Initialize the transform wrapper.

        Args:
            albumentations_transform (Callable): Albumentations callable returning a dict with an ``image`` key.
        """
        self.albumentations_transform = albumentations_transform

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a single-channel tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape [1, H, W].

        Returns:
            torch.Tensor: Augmented tensor with preserved shape.
        """
        img_np = tensor.squeeze(0).numpy()
        augmented = self.albumentations_transform(image=img_np)
        return torch.from_numpy(augmented["image"][None, ...])


class AlbumentationsPairedChannels:
    """Apply an albumentations transform to paired tensors (target/condition)."""

    def __init__(self, albumentations_transform: Callable):
        """Initialize the paired transform wrapper.

        Args:
            albumentations_transform (Callable): Albumentations callable returning ``image`` and
                ``condition_image`` keys.
        """
        self.albumentations_transform = albumentations_transform

    def __call__(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Augment both images within the provided dict.

        Args:
            data (dict): Dictionary with ``image`` and ``condition_image`` tensors.

        Returns:
            dict: Dictionary containing the augmented tensors.
        """
        img = data["image"].squeeze(0).numpy()
        cond = data["condition_image"].squeeze(0).numpy()
        augmented = self.albumentations_transform(image=img, condition_image=cond)
        data["image"] = torch.from_numpy(augmented["image"][None, ...])
        data["condition_image"] = torch.from_numpy(augmented["condition_image"][None, ...])
        return data


class DatasetWithAttributes:
    """Wrap a base dataset to attach per-sample attributes."""

    def __init__(self, base_dataset: Dataset, attributes: list[dict[str, float]]):
        """Initialize the wrapper dataset.

        Args:
            base_dataset: Underlying dataset returning image tensors.
            attributes: List of attribute dictionaries aligned with the dataset order.
        """
        self.base_dataset = base_dataset
        self.attributes = attributes

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, float]]:
        """Return image tensor and its attributes."""
        image = self.base_dataset[index]
        return image, self.attributes[index]


def collate_with_attributes(
    batch: list[tuple[torch.Tensor, dict[str, float]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Custom collate to stack images and group attributes."""
    images = torch.stack([item[0] for item in batch], dim=0)
    attribute_keys = batch[0][1].keys()
    attributes = {
        key: torch.tensor([float(item[1][key]) for item in batch], dtype=torch.float32) for key in attribute_keys
    }
    return images, attributes


def _load_attribute_json(attribute_file: str) -> dict[str, dict[str, float]]:
    """Load attribute JSON mapping filenames to attribute dictionaries.

    Args:
        attribute_file: Path to a JSON file.

    Returns:
        Mapping of filename to {attribute_name: value}.

    Raises:
        FileNotFoundError: If the JSON file is missing.
        ValueError: If the JSON cannot be parsed.
    """
    if not os.path.exists(attribute_file):
        raise FileNotFoundError(f"Attribute file not found: {attribute_file}")

    try:
        with open(attribute_file, encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid attribute JSON: {attribute_file}") from exc


def _select_attribute_sources(attribute_file: str | dict[str, str], data_source: str) -> dict[str, dict[str, float]]:
    """Select attribute mappings depending on the data source."""
    if isinstance(attribute_file, str):
        return {data_source: _load_attribute_json(attribute_file)}

    if isinstance(attribute_file, dict):
        mappings: dict[str, dict[str, float]] = {}
        for source, path in attribute_file.items():
            mappings[source] = _load_attribute_json(path)
        return mappings

    raise ValueError("regularized_attributes.attribute_file must be a string or mapping from source to file.")


def _normalize_attributes(
    attributes: dict[str, float],
    normalize_cfg: dict[str, Any] | None,
) -> dict[str, float]:
    """Normalize attribute values if requested."""
    if not normalize_cfg:
        return attributes

    if not normalize_cfg.get("enabled", False):
        return attributes

    divisor = float(normalize_cfg.get("divisor", 1.0))
    if divisor == 0:
        raise ValueError("Normalization divisor must be non-zero.")

    return {key: float(value) / divisor for key, value in attributes.items()}


def _filter_attributes_for_paths(
    paths: list[str],
    attribute_sources: dict[str, dict[str, float]],
    attribute_latent_mapping: dict[str, Any],
    normalize_cfg: dict[str, Any] | None,
) -> list[dict[str, float]]:
    """Extract and normalize attributes for a list of image paths.

    Args:
        paths (list[str]): Image file paths.
        attribute_sources (dict[str, dict[str, float]]): Per-source attribute JSONs keyed by source name.
        attribute_latent_mapping (dict[str, Any]): Attribute config kept for AR-VAE.
        normalize_cfg (dict[str, Any] | None): Optional normalization configuration.

    Returns:
        list[dict[str, float]]: Attributes aligned with ``paths`` order.

    Raises:
        FileNotFoundError: If an attribute entry is missing for a given path.
        KeyError: If expected attribute keys are missing.
        ValueError: If the data source cannot be inferred from the path.
    """
    attributes: list[dict[str, float]] = []
    for path in paths:
        base = os.path.basename(path)
        if "edente" in path:
            source_key = "edente"
        elif "dente" in path:
            source_key = "dente"
        else:
            raise ValueError(f"Cannot identify data source from path: {path}")

        mapping = attribute_sources.get(source_key, {})
        attribute_dict = mapping.get(base)
        if attribute_dict is None:
            raise FileNotFoundError(f"Attribute entry missing for {base} in source {source_key}")

        filtered = {key: float(attribute_dict[key]) for key in attribute_latent_mapping if key in attribute_dict}
        if len(filtered) != len(attribute_latent_mapping):
            missing = set(attribute_latent_mapping).difference(filtered)
            raise KeyError(f"Missing attributes for {base}: {missing}")

        filtered = _normalize_attributes(filtered, normalize_cfg)
        attributes.append(filtered)
    return attributes


def _print_dataset_stats(
    train_ds: Dataset,
    val_ds: Dataset,
    data_source: str,
    train_split: float,
    val_dir: str | None,
) -> None:
    """Print detailed dataset statistics.

    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        data_source: Data source name
        train_split: Train/val split ratio
        val_dir: Optional validation directory
    """
    total = len(train_ds) + len(val_ds)

    print(f"\n{'=' * 60}")
    print("Dataset Statistics")
    print(f"{'=' * 60}")
    print(f"Data source: {data_source}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Total: {total}")

    if val_dir is None:
        print(f"Split ratio: {len(train_ds) / total:.1%} / {len(val_ds) / total:.1%}")
    else:
        print("Split: External validation directory")

    # Sample statistics
    sample = train_ds[0][0] if isinstance(train_ds[0], tuple) else train_ds[0]
    print("\nImage properties:")
    print(f"  Shape: {sample.shape}")
    print(f"  Dtype: {sample.dtype}")
    print(f"  Range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"  Mean: {sample.mean():.3f}")
    print(f"  Std: {sample.std():.3f}")
    print(f"{'=' * 60}\n")


def create_vae_inference_dataloader(
    input_dir: str,
    patch_size: tuple[int, int],
    batch_size: int,
    num_samples: int | None = None,
    num_workers: int = 4,
) -> tuple[DataLoader, list[str]]:
    """Create a single dataloader for VAE inference or evaluation.

    This reuses the same normalization (LocalNormalizeByMask) and resizing
    as the training pipeline but does not apply augmentation, caching,
    attributes, or splitting.

    Args:
        input_dir: Directory containing .tif images.
        patch_size: Spatial resize target (H, W).
        batch_size: Batch size for iteration.
        num_samples: Optional cap on number of images to process.
        num_workers: Number of dataloader workers.

    Returns:
        Tuple of (dataloader, list of image paths).

    Raises:
        FileNotFoundError: If no .tif images are found in ``input_dir``.
    """
    tif_paths = sorted(glob(os.path.join(input_dir, "*.tif")))
    if len(tif_paths) == 0:
        raise FileNotFoundError(f"No .tif images found in {input_dir}")
    if num_samples is not None:
        tif_paths = tif_paths[:num_samples]

    transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(patch_size),
            LocalNormalizeByMask(),
            EnsureType(dtype=torch.float32),
        ]
    )
    dataset = Dataset(data=tif_paths, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader, tif_paths


def create_vae_dataloaders(
    data_base_dir: str,
    batch_size: int,
    patch_size: tuple[int, int],
    augment: bool = False,
    rank: int = 0,
    data_source: str = "edente",
    train_split: float = 0.9,
    num_workers: int = 4,
    seed: int | None = 42,
    subset_size: int | None = None,
    val_dir: str | None = None,
    cache_rate: float = 0.0,
    distributed: bool = False,
    world_size: int = 1,
    ar_vae_enabled: bool = False,
    regularized_attributes: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """Create train and validation dataloaders for VAE training.

    Args:
        data_base_dir: Base directory containing image subfolders
        batch_size: Batch size for dataloaders
        patch_size: Target image size (H, W)
        augment: Whether to apply data augmentation
        rank: Rank of current process (for printing info on rank 0)
        data_source: Which images to load - "edente", "dente", or "both"
        train_split: Train/val split ratio (default: 0.9 = 90% train, 10% val)
        num_workers: Number of worker processes for data loading (default: 4)
        seed: Random seed for reproducibility (default: 42, None = no seed)
        subset_size: Use only first N images for debugging (default: None = all)
        val_dir: Optional separate validation directory (default: None = use split)
        cache_rate: Fraction of dataset to cache in RAM, 0.0 to 1.0 (default: 0.0)
        distributed: Use DistributedSampler for DDP training (default: False)
        world_size: Number of processes for DDP (default: 1)
        ar_vae_enabled: Whether AR-VAE is enabled (default: False)
        regularized_attributes: Attribute configuration block for AR-VAE
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Tuple of (train_loader, val_loader, train_paths, val_paths)

    Raises:
        ValueError: If data_source is invalid, train_split is not in [0, 1], or cache_rate is not in [0, 1]
        FileNotFoundError: If no .tif images are found in the specified directories
    """
    # Validate parameters
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(f"cache_rate must be in [0, 1], got {cache_rate}")

    # Load images based on data_source and align attributes if requested
    if data_source == "edente":
        data_dir = os.path.join(data_base_dir, "edente")
        tif_paths = sorted(glob(os.path.join(data_dir, "*.tif")))
    elif data_source == "dente":
        data_dir = os.path.join(data_base_dir, "dente")
        tif_paths = sorted(glob(os.path.join(data_dir, "*.tif")))
    elif data_source == "both":
        dir_edente = os.path.join(data_base_dir, "edente")
        dir_dente = os.path.join(data_base_dir, "dente")
        tif_paths_edente = sorted(glob(os.path.join(dir_edente, "*.tif")))
        tif_paths_dente = sorted(glob(os.path.join(dir_dente, "*.tif")))
        tif_paths = tif_paths_edente + tif_paths_dente
    else:
        raise ValueError(f"data_source must be 'edente', 'dente', or 'both', got '{data_source}'")

    if len(tif_paths) == 0:
        raise FileNotFoundError(f"Aucune image .tif trouvée dans {data_base_dir}/{data_source}")

    # Apply subset for debugging
    if subset_size is not None:
        tif_paths = tif_paths[:subset_size]
        if rank == 0:
            print(f"⚠️  Using subset of {subset_size} images for debugging")

    attributes_per_image: list[dict[str, float]] | None = None
    train_attributes: list[dict[str, float]] | None = None
    val_attributes: list[dict[str, float]] | None = None

    if ar_vae_enabled:
        if regularized_attributes is None:
            raise ValueError("AR-VAE enabled but regularized_attributes config is missing.")

        attribute_file_cfg = regularized_attributes.get("attribute_file")
        raw_mapping = regularized_attributes.get("attribute_latent_mapping", {})
        attribute_latent_mapping = {k: v for k, v in raw_mapping.items() if not str(k).startswith("_")}
        if not attribute_latent_mapping:
            raise ValueError("attribute_latent_mapping must be provided when AR-VAE is enabled.")

        attribute_sources = _select_attribute_sources(attribute_file_cfg, data_source)

        attributes_per_image = []
        normalize_cfg = regularized_attributes.get("normalize_attributes")

        if data_source == "both":
            attributes_per_image = _filter_attributes_for_paths(
                paths=tif_paths,
                attribute_sources=attribute_sources,
                attribute_latent_mapping=attribute_latent_mapping,
                normalize_cfg=normalize_cfg,
            )
        else:
            source_key = data_source
            mapping = attribute_sources.get(source_key)
            if mapping is None:
                raise ValueError(f"No attribute mapping found for source {source_key}")

            attributes_per_image = _filter_attributes_for_paths(
                paths=tif_paths,
                attribute_sources=attribute_sources,
                attribute_latent_mapping=attribute_latent_mapping,
                normalize_cfg=normalize_cfg,
            )

    # Shuffle with seed for reproducibility
    if seed is not None:
        random.seed(seed)
        if attributes_per_image is not None:
            paired = list(zip(tif_paths, attributes_per_image, strict=True))
            random.shuffle(paired)
            tif_paths, attributes_per_image = zip(*paired, strict=False)
            tif_paths = list(tif_paths)
            attributes_per_image = list(attributes_per_image)
        else:
            tif_paths_copy = tif_paths.copy()
            random.shuffle(tif_paths_copy)
            tif_paths = tif_paths_copy

        # Handle validation directory
        if val_dir is not None:
            # Use separate validation directory
            if data_source == "edente":
                val_data_dir = os.path.join(val_dir, "edente")
            elif data_source == "dente":
                val_data_dir = os.path.join(val_dir, "dente")
            elif data_source == "both":
                val_dir_edente = os.path.join(val_dir, "edente")
                val_dir_dente = os.path.join(val_dir, "dente")
                val_paths_edente = sorted(glob(os.path.join(val_dir_edente, "*.tif")))
                val_paths_dente = sorted(glob(os.path.join(val_dir_dente, "*.tif")))
                val_paths = val_paths_edente + val_paths_dente
            else:
                raise ValueError(f"Invalid data_source: {data_source}")

            if data_source != "both":
                val_paths = sorted(glob(os.path.join(val_data_dir, "*.tif")))

            if len(val_paths) == 0:
                raise FileNotFoundError(f"Aucune image .tif trouvée dans {val_dir}/{data_source}")

            train_paths = tif_paths  # Use all data for training
            if attributes_per_image is not None:
                train_attributes = attributes_per_image
                val_attributes = []
                normalize_cfg = regularized_attributes.get("normalize_attributes") if regularized_attributes else None
                raw_mapping = (
                    regularized_attributes.get("attribute_latent_mapping", {}) if regularized_attributes else {}
                )
                attribute_latent_mapping = {k: v for k, v in raw_mapping.items() if not str(k).startswith("_")}

                val_attributes = _filter_attributes_for_paths(
                    paths=list(val_paths),
                    attribute_sources=attribute_sources,
                    attribute_latent_mapping=attribute_latent_mapping,
                    normalize_cfg=normalize_cfg,
                )
            if rank == 0:
                print(f"📁 Using external validation directory: {val_dir}")
        else:
            # Standard split
            split_idx = int(train_split * len(tif_paths))
            train_paths = tif_paths[:split_idx]
            val_paths = tif_paths[split_idx:]
            if attributes_per_image is not None:
                train_attributes = attributes_per_image[:split_idx]
                val_attributes = attributes_per_image[split_idx:]

    # Handle augmentation
    if augment:
        from .augmentation import get_albumentations_transform

        albumentations_transform = get_albumentations_transform()
        aug_monai = AlbumentationsSingleChannel(albumentations_transform)
    else:
        aug_monai = IdentityTensorTransform()

    # Define transforms
    transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(patch_size),
            LocalNormalizeByMask(),
            aug_monai,
            EnsureType(dtype=torch.float32),
        ]
    )

    # Create datasets with optional caching
    if cache_rate > 0:
        from monai.data import CacheDataset

        train_base = CacheDataset(
            data=train_paths, transform=transforms, cache_rate=cache_rate, num_workers=num_workers
        )
        val_base = CacheDataset(data=val_paths, transform=transforms, cache_rate=1.0, num_workers=num_workers)
        if rank == 0:
            print(f"🚀 Caching {cache_rate * 100:.0f}% of training data in RAM")
    else:
        train_base = Dataset(data=train_paths, transform=transforms)
        val_base = Dataset(data=val_paths, transform=transforms)

    if ar_vae_enabled:
        if train_attributes is None or val_attributes is None:
            raise ValueError("Attributes must be available when AR-VAE is enabled.")
        train_ds = DatasetWithAttributes(train_base, list(train_attributes))
        val_ds = DatasetWithAttributes(val_base, list(val_attributes))
    else:
        train_ds = train_base
        val_ds = val_base

    # Create dataloaders with optional distributed sampling
    collate_fn = collate_with_attributes if ar_vae_enabled else None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed if seed is not None else 0
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=seed if seed is not None else 0
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        if rank == 0:
            print(f"⚡ Using DistributedSampler for {world_size} GPUs")
    else:
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

    # Print dataset statistics
    if rank == 0:
        _print_dataset_stats(train_ds, val_ds, data_source, train_split, val_dir)

    return train_loader, val_loader, train_paths, val_paths


def create_ldm_dataloaders(
    data_base_dir: str,
    batch_size: int,
    patch_size: tuple[int, int],
    augment: bool = False,
    rank: int = 0,
    target: str = "edente",
    condition: str = "dente",
    train_split: float = 0.9,
    num_workers: int = 4,
    seed: int | None = 42,
    subset_size: int | None = None,
    cache_rate: float = 0.0,
    distributed: bool = False,
    world_size: int = 1,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for LDM training.

    Loads paired images from target and condition folders.

    Args:
        data_base_dir: Base directory containing image subfolders
        batch_size: Batch size for dataloaders
        patch_size: Target image size (H, W)
        augment: Whether to apply data augmentation
        rank: Rank of current process (for printing info on rank 0)
        target: Target image folder (default: "edente")
        condition: Condition image folder (default: "dente")
        train_split: Train/val split ratio (default: 0.9 = 90% train, 10% val)
        num_workers: Number of worker processes for data loading (default: 4)
        seed: Random seed for reproducibility (default: 42, None = no seed)
        subset_size: Use only first N pairs for debugging (default: None = all)
        cache_rate: Fraction of dataset to cache in RAM, 0.0 to 1.0 (default: 0.0)
        distributed: Use DistributedSampler for DDP training (default: False)
        world_size: Number of processes for DDP (default: 1)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Tuple of (train_loader, val_loader)
        Each batch returns (target_image, condition_image) tuple

    Raises:
        ValueError: If train_split is not in (0, 1) or cache_rate is not in [0, 1]
        FileNotFoundError: If no .tif images are found in target or condition folders
    """
    # Validate parameters
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(f"cache_rate must be in [0, 1], got {cache_rate}")

    dir_target = os.path.join(data_base_dir, target)
    dir_condition = os.path.join(data_base_dir, condition)

    tif_paths_target = sorted(glob(os.path.join(dir_target, "*.tif")))
    tif_paths_condition = sorted(glob(os.path.join(dir_condition, "*.tif")))

    if len(tif_paths_target) == 0 or len(tif_paths_condition) == 0:
        raise FileNotFoundError(f"Aucune image trouvée dans {dir_target} ou {dir_condition}")
    if len(tif_paths_target) != len(tif_paths_condition):
        raise ValueError(
            f"Les dossiers {target} et {condition} doivent contenir le même nombre d'images. "
            f"Trouvé: {len(tif_paths_target)} vs {len(tif_paths_condition)}"
        )

    # Create paired data
    paired_data = [
        {"image": t, "condition_image": c} for t, c in zip(tif_paths_target, tif_paths_condition, strict=True)
    ]

    # Apply subset for debugging
    if subset_size is not None:
        paired_data = paired_data[:subset_size]
        if rank == 0:
            print(f"⚠️  Using subset of {subset_size} pairs for debugging")

    # Shuffle with seed for reproducibility
    if seed is not None:
        random.seed(seed)
        paired_data_copy = paired_data.copy()
        random.shuffle(paired_data_copy)
        paired_data = paired_data_copy

    # Split train/val
    split_idx = int(train_split * len(paired_data))
    train_data = paired_data[:split_idx]
    val_data = paired_data[split_idx:]

    # Handle augmentation
    if augment:
        from .augmentation import get_albumentations_transform

        albumentations_transform = get_albumentations_transform()
        aug_monai = AlbumentationsPairedChannels(albumentations_transform)
    else:
        aug_monai = IdentityTensorTransform()

    # Define transforms (ALWAYS resize to patch_size)
    transform_list = [
        LoadImaged(keys=["image", "condition_image"]),
        EnsureChannelFirstd(keys=["image", "condition_image"]),
        ResizeD(keys=["image", "condition_image"], spatial_size=patch_size),  # Always resize
        EnsureTyped(keys=["image", "condition_image"], dtype=torch.float32),
        ApplyLocalNormd(keys=["image", "condition_image"]),
        aug_monai,  # Apply augmentation if enabled
        ToTuple(keys=["image", "condition_image"]),
    ]

    transforms = Compose(transform_list)

    # Create datasets with optional caching
    if cache_rate > 0:
        from monai.data import CacheDataset

        train_ds = CacheDataset(data=train_data, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
        val_ds = CacheDataset(data=val_data, transform=transforms, cache_rate=1.0, num_workers=num_workers)
        if rank == 0:
            print(f"🚀 Caching {cache_rate * 100:.0f}% of training pairs in RAM")
    else:
        train_ds = Dataset(data=train_data, transform=transforms)
        val_ds = Dataset(data=val_data, transform=transforms)

    # Create dataloaders with optional distributed sampling
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed if seed is not None else 0
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=seed if seed is not None else 0
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True
        )

        if rank == 0:
            print(f"⚡ Using DistributedSampler for {world_size} GPUs")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Print dataset statistics
    if rank == 0:
        sample = next(iter(train_loader))
        total = len(train_ds) + len(val_ds)

        print(f"\n{'=' * 60}")
        print("LDM Dataset Statistics")
        print(f"{'=' * 60}")
        print(f"Target: {target} | Condition: {condition}")
        print(f"Train pairs: {len(train_ds)}")
        print(f"Val pairs: {len(val_ds)}")
        print(f"Total: {total}")
        print(f"Split ratio: {len(train_ds) / total:.1%} / {len(val_ds) / total:.1%}")
        print("\nImage shapes:")
        print(f"  Target: {sample[0].shape}")
        print(f"  Condition: {sample[1].shape}")
        print(f"{'=' * 60}\n")

    return train_loader, val_loader
