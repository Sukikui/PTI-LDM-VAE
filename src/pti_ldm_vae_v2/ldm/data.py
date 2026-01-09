from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

import torch
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureChannelFirstd,
    EnsureType,
    EnsureTyped,
    LoadImage,
    LoadImaged,
)

from pti_ldm_vae_v2.vae_regression_common import DEFAULT_NUM_WORKERS, LocalNormalizeByMask, list_tif_paths


class ApplyLocalNormalizeDict:
    """Apply LocalNormalizeByMask to a dictionary of images.

    Args:
        keys (list[str]): Dictionary keys to normalize.
    """

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        self.normalizer = LocalNormalizeByMask()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply normalization to the configured keys.

        Args:
            data (dict[str, Any]): Dictionary containing image tensors.

        Returns:
            dict[str, Any]: Updated dictionary with normalized tensors.
        """
        for key in self.keys:
            data[key] = torch.as_tensor(self.normalizer(data[key]))
        return data


class PairToTuple:
    """Transform dict {'edentulous', 'dentate'} to tuple (edentulous, dentate)."""

    def __call__(self, data: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert dict inputs to a tuple.

        Args:
            data (dict[str, torch.Tensor]): Dictionary containing paired tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (edentulous, dentate) tensors.
        """
        return data["edentulous"], data["dentate"]


def _pair_paths(edente_paths: list[str], dente_paths: list[str]) -> list[dict[str, str]]:
    """Create paired path dictionaries and validate lengths.

    Args:
        edente_paths (list[str]): Sorted list of edentulous image paths.
        dente_paths (list[str]): Sorted list of dentate image paths.

    Returns:
        list[dict[str, str]]: Paired dictionaries with keys ``edentulous`` and ``dentate``.

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


def _build_pair_transform() -> Compose:
    """Build a MONAI transform pipeline for paired images.

    Returns:
        Compose: Composed transform producing a tuple (edentulous, dentate) without resizing.
    """
    return Compose(
        [
            LoadImaged(keys=["edentulous", "dentate"]),
            EnsureChannelFirstd(keys=["edentulous", "dentate"]),
            EnsureTyped(keys=["edentulous", "dentate"], dtype=torch.float32),
            ApplyLocalNormalizeDict(keys=["edentulous", "dentate"]),
            PairToTuple(),
        ]
    )


def _split_pairs(
    paired: list[dict[str, str]],
    train_split: float,
    val_dir: str | None,
    seed: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split paired data into train/val or use external validation directory.

    Args:
        paired (list[dict[str, str]]): Full list of paired path dictionaries.
        train_split (float): Train/val ratio.
        val_dir (str | None): Optional validation directory (overrides split).
        seed (int | None): Seed for shuffling pairs.

    Returns:
        tuple[list[dict[str, str]], list[dict[str, str]]]: Train pairs and validation pairs.
    """
    if seed is not None:
        random.seed(seed)
        random.shuffle(paired)

    if val_dir is not None:
        val_edente = list_tif_paths(val_dir, "edente")
        val_dente = list_tif_paths(val_dir, "dente")
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
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int | None = 42,
    subset_size: int | None = None,
    val_dir: str | None = None,
    cache_rate: float = 0.0,
) -> tuple[DataLoader, DataLoader, list[dict[str, str]], list[dict[str, str]]]:
    """Create train/val dataloaders for paired dentate/edentulous images.

    Args:
        data_base_dir (str): Root directory containing ``edente/`` and ``dente/`` subfolders.
        batch_size (int): Batch size.
        patch_size (Iterable[int]): Unused (LDM input is not resized).
        train_split (float): Train/validation ratio when ``val_dir`` is not provided.
        num_workers (int): Number of dataloader workers.
        seed (int | None): Seed for shuffling; ``None`` disables shuffling.
        subset_size (int | None): Use only the first N paired samples for debugging.
        val_dir (str | None): Optional separate validation directory mirroring the same subfolder structure.
        cache_rate (float): Fraction of the training dataset to cache in RAM (0.0-1.0).

    Returns:
        tuple[DataLoader, DataLoader, list[dict[str, str]], list[dict[str, str]]]:
            Train loader, val loader, and the paired path lists.
    """
    if not 0 < train_split < 1:
        raise ValueError(f"train_split must be in (0, 1), got {train_split}")
    if not 0.0 <= cache_rate <= 1.0:
        raise ValueError(f"cache_rate must be in [0, 1], got {cache_rate}")

    edente_paths = list_tif_paths(data_base_dir, "edente")
    dente_paths = list_tif_paths(data_base_dir, "dente")
    paired = _pair_paths(edente_paths, dente_paths)

    if subset_size is not None:
        paired = paired[:subset_size]
        print(f"[INFO] Using subset of {subset_size} paired samples.")

    train_pairs, val_pairs = _split_pairs(
        paired=paired,
        train_split=train_split,
        val_dir=val_dir,
        seed=seed,
    )

    _ = patch_size
    transform = _build_pair_transform()
    if cache_rate > 0:
        train_base = CacheDataset(data=train_pairs, transform=transform, cache_rate=cache_rate, num_workers=num_workers)
        val_base = CacheDataset(data=val_pairs, transform=transform, cache_rate=1.0, num_workers=num_workers)
        print(f"[INFO] Caching {cache_rate * 100:.0f}% of training pairs in RAM.")
    else:
        train_base = Dataset(data=train_pairs, transform=transform)
        val_base = Dataset(data=val_pairs, transform=transform)

    collate_fn = list_data_collate
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

    total = len(train_base) + len(val_base)
    print("\n" + "=" * 60)
    print("LDM Paired Dataset Statistics")
    print("=" * 60)
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
    print("=" * 60 + "\n")

    return train_loader, val_loader, train_pairs, val_pairs


def build_ldm_inference_transform() -> Compose:
    """Build an inference transform pipeline for dentate images.

    Returns:
        Compose: Transform that loads, normalizes, and converts to float tensors.
    """
    return Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            LocalNormalizeByMask(),
            EnsureType(dtype=torch.float32),
        ]
    )


def create_ldm_inference_dataloader(
    *,
    input_dir: str,
    batch_size: int,
    num_samples: int | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> tuple[DataLoader, list[str]]:
    """Create a dataloader for LDM sampling without resizing inputs.

    Args:
        input_dir (str): Directory containing dentate images.
        batch_size (int): Batch size.
        num_samples (int | None): Optional cap on number of images.
        num_workers (int): Number of dataloader workers.

    Returns:
        tuple[DataLoader, list[str]]: Dataloader and list of image paths.
    """
    tif_paths = list_tif_paths(input_dir, data_source="dente")
    if num_samples is not None:
        tif_paths = tif_paths[:num_samples]

    transform = build_ldm_inference_transform()
    dataset = Dataset(data=tif_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader, tif_paths
