import json
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, Lambda, LoadImage, Resize

from .transforms import LocalNormalizeByMask, TifReader


def _list_tif_paths(data_base_dir: str, data_source: str) -> list[str]:
    """List .tif image paths for the requested data source.

    Args:
        data_base_dir (str): Root directory containing ``edente`` and/or ``dente`` subfolders.
        data_source (str): Either ``edente``, ``dente``, or ``both``.

    Returns:
        list[str]: Sorted list of .tif image paths.

    Raises:
        ValueError: If ``data_source`` is invalid.
        FileNotFoundError: If no .tif files are found.
    """
    base_path = Path(data_base_dir)
    direct_tifs = sorted(base_path.glob("*.tif"))
    if direct_tifs:
        return [str(path) for path in direct_tifs]

    if data_source == "edente":
        tif_paths = sorted((base_path / "edente").glob("*.tif"))
    elif data_source == "dente":
        tif_paths = sorted((base_path / "dente").glob("*.tif"))
    elif data_source == "both":
        tif_paths_edente = sorted((base_path / "edente").glob("*.tif"))
        tif_paths_dente = sorted((base_path / "dente").glob("*.tif"))
        tif_paths = tif_paths_edente + tif_paths_dente
    else:
        raise ValueError(f"data_source must be 'edente', 'dente', or 'both', got '{data_source}'")

    if len(tif_paths) == 0:
        raise FileNotFoundError(f"Aucune image .tif trouvée dans {data_base_dir}/{data_source}")
    return [str(path) for path in tif_paths]


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


class DatasetWithTargets:
    """Wrap a base dataset to attach vector targets."""

    def __init__(self, base_dataset: Dataset, targets: list[torch.Tensor]):
        """Initialize the dataset wrapper.

        Args:
            base_dataset (Dataset): Underlying dataset returning images.
            targets (list[torch.Tensor]): Target vectors aligned with the dataset order.
        """
        self.base_dataset = base_dataset
        self.targets = [target.clone() for target in targets]

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image tensor and its target vector."""
        image = self.base_dataset[index]
        return image, self.targets[index]

    def apply_target_transform(self, transform: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Apply a transform to all stored target vectors in-place.

        Args:
            transform (Callable[[torch.Tensor], torch.Tensor]): Transform applied to each target vector.
        """
        self.targets = [transform(target) for target in self.targets]

    def stacked_targets(self) -> torch.Tensor:
        """Return all targets stacked into a single tensor."""
        return torch.stack(self.targets, dim=0)


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


def _attributes_to_tensor(attributes: list[dict[str, float]], targets: list[str]) -> list[torch.Tensor]:
    """Convert attribute dicts to ordered target vectors.

    Args:
        attributes (list[dict[str, float]]): Attributes per image keyed by target name.
        targets (list[str]): Target names defining vector order.

    Returns:
        list[torch.Tensor]: Float tensors with shape [len(targets)].

    Raises:
        KeyError: If a target is missing in an attribute dict.
    """
    vectors: list[torch.Tensor] = []
    for attribute_dict in attributes:
        try:
            values = [float(attribute_dict[target]) for target in targets]
        except KeyError as exc:
            raise KeyError(f"Missing target {exc} in attributes.") from exc
        vectors.append(torch.tensor(values, dtype=torch.float32))
    return vectors


def build_vae_preprocess_transform(
    patch_size: tuple[int, int],
    *,
    use_tif_reader: bool = False,
) -> Compose:
    """Create a reusable VAE preprocessing pipeline.

    Args:
        patch_size: Target spatial size (height, width).
        use_tif_reader: When ``True``, use the custom TIF reader (analysis); otherwise rely on MONAI's LoadImage.

    Returns:
        Composed MONAI transform for loading, resizing, normalizing, and typing images.
    """
    if use_tif_reader:
        loader = Lambda(func=TifReader())
        channel_first = EnsureChannelFirst(channel_dim="no_channel")
    else:
        loader = LoadImage(image_only=True)
        channel_first = EnsureChannelFirst()

    return Compose(
        [
            loader,
            channel_first,
            Resize(patch_size),
            LocalNormalizeByMask(),
            EnsureType(dtype=torch.float32),
        ]
    )


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


def _build_training_transform(patch_size: tuple[int, int]) -> Compose:
    """Construct the training transform pipeline (sans augmentation)."""
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Resize(patch_size),
            LocalNormalizeByMask(),
            EnsureType(dtype=torch.float32),
        ]
    )


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
    tif_paths = _list_tif_paths(input_dir, data_source="both")
    if len(tif_paths) == 0:
        raise FileNotFoundError(f"No .tif images found in {input_dir}")
    if num_samples is not None:
        tif_paths = tif_paths[:num_samples]

    transforms = build_vae_preprocess_transform(patch_size)
    dataset = Dataset(data=tif_paths, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader, tif_paths


def create_vae_dataloaders(
    data_base_dir: str,
    batch_size: int,
    patch_size: tuple[int, int],
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
    tif_paths = _list_tif_paths(data_base_dir, data_source)

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
            val_paths = _list_tif_paths(val_dir, data_source)

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

    transforms = _build_training_transform(patch_size)

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
    collate_fn = collate_with_attributes if ar_vae_enabled else list_data_collate
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


def create_regression_dataloaders(
    data_base_dir: str,
    attributes_path: str | dict[str, str],
    targets: list[str],
    batch_size: int,
    patch_size: tuple[int, int],
    train_split: float = 0.9,
    num_workers: int = 4,
    seed: int | None = 42,
    subset_size: int | None = None,
    val_dir: str | None = None,
    cache_rate: float = 0.0,
    data_source: str = "edente",
    normalize_attributes: dict[str, Any] | None = None,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """Create train and validation loaders for the latent regression head.

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
        rank (int): Process rank (for logging).

    Returns:
        tuple[DataLoader, DataLoader, list[str], list[str]]: Train loader, val loader, and corresponding file lists.
    """
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(f"cache_rate must be in [0, 1], got {cache_rate}")
    if len(targets) == 0:
        raise ValueError("targets must contain at least one entry.")

    tif_paths = _list_tif_paths(data_base_dir, data_source)
    if subset_size is not None:
        tif_paths = tif_paths[:subset_size]
        if rank == 0:
            print(f"⚠️  Using subset of {subset_size} images for debugging")

    attribute_sources = _select_attribute_sources(attributes_path, data_source)
    attribute_mapping = {target: target for target in targets}
    attributes_per_image = _filter_attributes_for_paths(
        paths=tif_paths,
        attribute_sources=attribute_sources,
        attribute_latent_mapping=attribute_mapping,
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
        val_paths = _list_tif_paths(val_dir, data_source)
        train_paths = tif_paths
        val_attributes = _filter_attributes_for_paths(
            paths=list(val_paths),
            attribute_sources=attribute_sources,
            attribute_latent_mapping=attribute_mapping,
            normalize_cfg=normalize_attributes,
        )
        train_attributes = attributes_per_image
        if rank == 0:
            print(f"📁 Using external validation directory: {val_dir}")
    else:
        split_idx = int(train_split * len(tif_paths))
        train_paths = tif_paths[:split_idx]
        val_paths = tif_paths[split_idx:]
        train_attributes = attributes_per_image[:split_idx]
        val_attributes = attributes_per_image[split_idx:]

    train_targets = _attributes_to_tensor(train_attributes, targets)
    val_targets = _attributes_to_tensor(val_attributes, targets)

    transforms = _build_training_transform(patch_size)
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

    if rank == 0:
        _print_dataset_stats(train_ds, val_ds, data_source, train_split, val_dir)

    return train_loader, val_loader, train_paths, val_paths


def create_regression_eval_dataloader(
    input_dir: str,
    attributes_path: str | dict[str, str],
    targets: list[str],
    patch_size: tuple[int, int],
    batch_size: int,
    num_workers: int = 4,
    num_samples: int | None = None,
    data_source: str = "edente",
    normalize_attributes: dict[str, Any] | None = None,
) -> tuple[DataLoader, list[str]]:
    """Build a dataloader for regression evaluation (images + targets).

    Args:
        input_dir (str): Directory containing images grouped by source (``edente``/``dente`` or mixed).
        attributes_path (str | dict[str, str]): JSON path or mapping of source → JSON path.
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
    tif_paths = _list_tif_paths(input_dir, data_source)
    if num_samples is not None:
        tif_paths = tif_paths[:num_samples]

    attribute_sources = _select_attribute_sources(attributes_path, data_source)
    attribute_mapping = {target: target for target in targets}
    attributes_per_image = _filter_attributes_for_paths(
        paths=tif_paths,
        attribute_sources=attribute_sources,
        attribute_latent_mapping=attribute_mapping,
        normalize_cfg=normalize_attributes,
    )

    target_tensors = _attributes_to_tensor(attributes_per_image, targets)
    transforms = build_vae_preprocess_transform(patch_size)
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
    num_workers: int = 4,
) -> tuple[DataLoader, list[str]]:
    """Dataloader for regression inference without targets."""
    return create_vae_inference_dataloader(
        input_dir=input_dir,
        patch_size=patch_size,
        batch_size=batch_size,
        num_samples=num_samples,
        num_workers=num_workers,
    )
