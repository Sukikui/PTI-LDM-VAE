import os
import random
from glob import glob

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
    sample = train_ds[0]
    print("\nImage properties:")
    print(f"  Shape: {sample.shape}")
    print(f"  Dtype: {sample.dtype}")
    print(f"  Range: [{sample.min():.3f}, {sample.max():.3f}]")
    print(f"  Mean: {sample.mean():.3f}")
    print(f"  Std: {sample.std():.3f}")
    print(f"{'=' * 60}\n")


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
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
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
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Tuple of (train_loader, val_loader)

    Raises:
        ValueError: If data_source is invalid, train_split is not in [0, 1], or cache_rate is not in [0, 1]
        FileNotFoundError: If no .tif images are found in the specified directories
    """
    # Validate parameters
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(f"cache_rate must be in [0, 1], got {cache_rate}")

    # Load images based on data_source
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

    # Shuffle with seed for reproducibility
    if seed is not None:
        random.seed(seed)
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
        if rank == 0:
            print(f"📁 Using external validation directory: {val_dir}")
    else:
        # Standard split
        split_idx = int(train_split * len(tif_paths))
        train_paths = tif_paths[:split_idx]
        val_paths = tif_paths[split_idx:]

    # Handle augmentation
    if augment:
        from .augmentation import get_albumentations_transform

        albumentations_transform = get_albumentations_transform()

        class AugAlb:
            def __call__(self, img):
                img_np = img.squeeze(0).numpy()
                aug = albumentations_transform(image=img_np)
                return torch.from_numpy(aug["image"][None, ...])

        aug_monai = AugAlb()
    else:

        def aug_monai(x):
            return x

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

        train_ds = CacheDataset(data=train_paths, transform=transforms, cache_rate=cache_rate, num_workers=num_workers)
        val_ds = CacheDataset(data=val_paths, transform=transforms, cache_rate=1.0, num_workers=num_workers)
        if rank == 0:
            print(f"🚀 Caching {cache_rate * 100:.0f}% of training data in RAM")
    else:
        train_ds = Dataset(data=train_paths, transform=transforms)
        val_ds = Dataset(data=val_paths, transform=transforms)

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
        _print_dataset_stats(train_ds, val_ds, data_source, train_split, val_dir)

    return train_loader, val_loader


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

        class AugAlb:
            def __call__(self, data):
                img = data["image"].squeeze(0).numpy()
                cond = data["condition_image"].squeeze(0).numpy()
                aug = albumentations_transform(image=img, condition_image=cond)
                data["image"] = torch.from_numpy(aug["image"][None, ...])
                data["condition_image"] = torch.from_numpy(aug["condition_image"][None, ...])
                return data

        aug_monai = AugAlb()
    else:

        def aug_monai(x):
            return x

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
