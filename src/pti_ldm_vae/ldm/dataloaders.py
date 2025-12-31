from __future__ import annotations

import random
from collections.abc import Iterable

import torch
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Resized

from pti_ldm_vae.data.dataloaders import _list_tif_paths
from pti_ldm_vae.data.transforms import ApplyLocalNormd


class PairToTuple:
    """Transform dict {'edentulous', 'dentate'} to tuple (edentulous, dentate)."""

    def __call__(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return data["edentulous"], data["dentate"]


def _pair_paths(edente_paths: list[str], dente_paths: list[str]) -> list[dict[str, str]]:
    """Create paired path dictionaries and validate lengths.

    Args:
        edente_paths: Sorted list of edentulous image paths.
        dente_paths: Sorted list of dentate image paths.

    Returns:
        List of dicts with keys ``edentulous`` and ``dentate``.

    Raises:
        FileNotFoundError: If either list is empty.
        ValueError: If dataset lengths differ.
    """
    if len(edente_paths) == 0 or len(dente_paths) == 0:
        raise FileNotFoundError("No .tif images found in edente/ or dente/ subfolders.")
    if len(edente_paths) != len(dente_paths):
        raise ValueError(
            f"Mismatched paired datasets: {len(edente_paths)} edente images vs {len(dente_paths)} dente images."
        )
    return [{"edentulous": e, "dentate": d} for e, d in zip(edente_paths, dente_paths, strict=True)]


def _build_pair_transform(patch_size: tuple[int, int]) -> Compose:
    """Build a MONAI transform pipeline for paired images.

    Args:
        patch_size: Target spatial size (H, W).

    Returns:
        Composed transform producing a tuple (edentulous, dentate).
    """
    return Compose(
        [
            LoadImaged(keys=["edentulous", "dentate"]),
            EnsureChannelFirstd(keys=["edentulous", "dentate"]),
            Resized(keys=["edentulous", "dentate"], spatial_size=patch_size),
            EnsureTyped(keys=["edentulous", "dentate"], dtype=torch.float32),
            ApplyLocalNormd(keys=["edentulous", "dentate"]),
            PairToTuple(),
        ]
    )


def _split_pairs(
    paired: list[dict[str, str]],
    train_split: float,
    val_dir: str | None,
    data_source: str,
    seed: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split paired data into train/val or use external validation directory.

    Args:
        paired: Full list of paired path dictionaries.
        train_split: Train/val ratio.
        val_dir: Optional validation directory (overrides split).
        data_source: Unused placeholder for parity with other loaders.
        seed: Seed for shuffling pairs.

    Returns:
        Train pairs and validation pairs.
    """
    if seed is not None:
        random.seed(seed)
        random.shuffle(paired)

    if val_dir is not None:
        val_edente = _list_tif_paths(val_dir, "edente")
        val_dente = _list_tif_paths(val_dir, "dente")
        val_pairs = _pair_paths(val_edente, val_dente)
        train_pairs = paired
    else:
        split_idx = int(train_split * len(paired))
        train_pairs = paired[:split_idx]
        val_pairs = paired[split_idx:]
    return train_pairs, val_pairs


def create_ldm_dataloaders(
    *,
    data_base_dir: str,
    batch_size: int,
    patch_size: Iterable[int],
    train_split: float = 0.9,
    num_workers: int = 4,
    seed: int | None = 42,
    subset_size: int | None = None,
    val_dir: str | None = None,
    cache_rate: float = 0.0,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader, list[dict[str, str]], list[dict[str, str]]]:
    """Create train/val dataloaders for paired dentate/edentulous images.

    Args:
        data_base_dir: Root directory containing ``edente/`` and ``dente/`` subfolders.
        batch_size: Batch size per process.
        patch_size: Spatial size (H, W) used for resizing.
        train_split: Train/validation ratio when ``val_dir`` is not provided.
        num_workers: Number of dataloader workers.
        seed: Seed for shuffling; ``None`` disables shuffling.
        subset_size: Use only the first N paired samples for debugging.
        val_dir: Optional separate validation directory mirroring the same subfolder structure.
        cache_rate: Fraction of the training dataset to cache in RAM (0.0-1.0).
        distributed: Whether to attach DistributedSampler.
        world_size: Number of processes for DDP.
        rank: Process rank (used for logging).

    Returns:
        Tuple containing train loader, val loader, and the corresponding paired path lists.
    """
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(f"cache_rate must be in [0, 1], got {cache_rate}")

    edente_paths = _list_tif_paths(data_base_dir, "edente")
    dente_paths = _list_tif_paths(data_base_dir, "dente")
    paired = _pair_paths(edente_paths, dente_paths)

    if subset_size is not None:
        paired = paired[:subset_size]
        if rank == 0:
            print(f"⚠️  Using subset of {subset_size} paired samples for debugging.")

    train_pairs, val_pairs = _split_pairs(
        paired=paired, train_split=train_split, val_dir=val_dir, data_source="both", seed=seed
    )

    transform = _build_pair_transform(tuple(patch_size))
    if cache_rate > 0:
        train_base = CacheDataset(data=train_pairs, transform=transform, cache_rate=cache_rate, num_workers=num_workers)
        val_base = CacheDataset(data=val_pairs, transform=transform, cache_rate=1.0, num_workers=num_workers)
        if rank == 0:
            print(f"🚀 Caching {cache_rate * 100:.0f}% of training pairs in RAM")
    else:
        train_base = Dataset(data=train_pairs, transform=transform)
        val_base = Dataset(data=val_pairs, transform=transform)

    collate_fn = list_data_collate

    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(
            train_base, num_replicas=world_size, rank=rank, shuffle=True, seed=seed if seed is not None else 0
        )
        val_sampler = DistributedSampler(
            val_base, num_replicas=world_size, rank=rank, shuffle=False, seed=seed if seed is not None else 0
        )
        train_loader = DataLoader(
            train_base,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_base,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        if rank == 0:
            print(f"⚡ Using DistributedSampler for {world_size} processes")
    else:
        train_loader = DataLoader(
            train_base,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_base,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    if rank == 0:
        total = len(train_base) + len(val_base)
        print(f"\n{'=' * 60}")
        print("LDM Paired Dataset Statistics")
        print(f"{'=' * 60}")
        print(f"Train samples: {len(train_base)}")
        print(f"Val samples: {len(val_base)}")
        print(f"Total: {total}")
        if val_dir is None:
            print(f"Split ratio: {len(train_base) / total:.1%} / {len(val_base) / total:.1%}")
        else:
            print(f"External validation directory: {val_dir}")
        sample = train_base[0]
        print("\nSample shapes:")
        print(f"  edentulous: {sample[0].shape}")
        print(f"  dentate:    {sample[1].shape}")
        print(f"{'=' * 60}\n")

    return train_loader, val_loader, train_pairs, val_pairs
