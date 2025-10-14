from glob import glob
import os
from typing import Tuple

import torch
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Resize,
    EnsureType,
    LoadImaged,
    EnsureChannelFirstd,
    ResizeD,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader

from .transforms import LocalNormalizeByMask, ApplyLocalNormd, ToTuple


def create_vae_dataloaders(
    data_base_dir: str,
    batch_size: int,
    patch_size: Tuple[int, int],
    augment: bool = False,
    rank: int = 0,
    data_source: str = "edente",
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for VAE training.

    Args:
        data_base_dir: Base directory containing image subfolders
        batch_size: Batch size for dataloaders
        patch_size: Target image size (H, W)
        augment: Whether to apply data augmentation
        rank: Rank of current process (for printing info on rank 0)
        data_source: Which images to load - "edente", "dente", or "both"
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Tuple of (train_loader, val_loader)
    """
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

    # Split train/val (90/10)
    split_idx = int(0.9 * len(tif_paths))
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
                return torch.from_numpy(aug['image'][None, ...])

        aug_monai = AugAlb()
    else:
        aug_monai = lambda x: x

    # Define transforms
    transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Resize(patch_size),
        LocalNormalizeByMask(),
        aug_monai,
        EnsureType(dtype=torch.float32),
    ])

    # Create datasets
    train_ds = Dataset(data=train_paths, transform=transforms)
    val_ds = Dataset(data=val_paths, transform=transforms)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if rank == 0:
        print(f"[VAE] Image shape {train_ds[0].shape}")
        print(f"[VAE] Data source: {data_source}")
        print(f"[VAE] Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    return train_loader, val_loader


def create_ldm_dataloaders(
    data_base_dir: str,
    batch_size: int,
    patch_size: Tuple[int, int],
    augment: bool = False,
    rank: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for LDM training.

    Loads paired images from "edente" (target) and "dente" (condition) folders.

    Args:
        data_base_dir: Base directory containing "edente" and "dente" subfolders
        batch_size: Batch size for dataloaders
        patch_size: Target image size (H, W)
        augment: Whether to apply data augmentation
        rank: Rank of current process (for printing info on rank 0)
        **kwargs: Additional arguments (for compatibility)

    Returns:
        Tuple of (train_loader, val_loader)
        Each batch returns (image, condition_image) tuple
    """
    dir_edente = os.path.join(data_base_dir, "edente")
    dir_dente = os.path.join(data_base_dir, "dente")

    tif_paths_edente = sorted(glob(os.path.join(dir_edente, "*.tif")))
    tif_paths_dente = sorted(glob(os.path.join(dir_dente, "*.tif")))

    if len(tif_paths_edente) == 0 or len(tif_paths_dente) == 0:
        raise FileNotFoundError(f"Aucune image trouvée dans {dir_edente} ou {dir_dente}")
    if len(tif_paths_edente) != len(tif_paths_dente):
        raise ValueError("Les dossiers denté et édenté doivent contenir le même nombre d'images.")

    # Create paired data
    paired_data = [
        {"image": e, "condition_image": d}
        for e, d in zip(tif_paths_edente, tif_paths_dente)
    ]

    # Split train/val (90/10)
    split_idx = int(0.9 * len(paired_data))
    train_data = paired_data[:split_idx]
    val_data = paired_data[split_idx:]

    # Handle augmentation
    if augment:
        from .augmentation import get_albumentations_transform
        albumentations_transform = get_albumentations_transform()

        class AugAlb:
            def __call__(self, data):
                img = data['image'].squeeze(0).numpy()
                cond = data['condition_image'].squeeze(0).numpy()
                aug = albumentations_transform(image=img, condition_image=cond)
                data['image'] = torch.from_numpy(aug['image'][None, ...])
                data['condition_image'] = torch.from_numpy(aug['condition_image'][None, ...])
                return data

        aug_monai = AugAlb()
    else:
        aug_monai = lambda x: x

    # Define transforms
    transform_list = [
        LoadImaged(keys=["image", "condition_image"]),
        EnsureChannelFirstd(keys=["image", "condition_image"]),
        EnsureTyped(keys=["image", "condition_image"], dtype=torch.float32),
        ApplyLocalNormd(keys=["image", "condition_image"]),
    ]

    # Add resize if augmentation is enabled
    if augment:
        transform_list.insert(2, ResizeD(keys=["image", "condition_image"], spatial_size=patch_size))

    transform_list.append(aug_monai)
    transform_list.append(ToTuple(keys=["image", "condition_image"]))

    transforms = Compose(transform_list)

    # Create datasets
    train_ds = Dataset(data=train_data, transform=transforms)
    val_ds = Dataset(data=val_data, transform=transforms)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    if rank == 0:
        sample = next(iter(train_loader))
        print(f"[LDM] Image shape: {sample[0].shape}, Condition shape: {sample[1].shape}")
        print(f"[LDM] Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    return train_loader, val_loader