# Data Module

Dataloader utilities for VAE training and evaluation.

## Overview

- **`create_vae_dataloaders`**: Single-image dataloaders for VAE training
- **`create_vae_inference_dataloader`**: Single dataloader for VAE inference/evaluation
- **`build_vae_preprocess_transform`**: Shared preprocessing pipeline (load/resize/normalize) reusable across scripts
- **Transform classes**: Custom preprocessing (normalization, masking)

Note: `create_vae_dataloaders` returns both loaders and file paths `(train_loader, val_loader, train_paths, val_paths)`. Use `_` to ignore paths if you only need the loaders. The VAE loaders now always apply an explicit collate function (`collate_with_attributes` for AR-VAE or MONAI's `list_data_collate` otherwise) to guarantee that batches come back as tensors/dicts instead of plain Python lists across Torch/MONAI versions; le code d'entraînement consolide aussi les batches reçus en listes (rare sur certains couples PyTorch/MONAI) avant l'envoi sur GPU/CPU.

______________________________________________________________________

## Quick Start

### VAE Dataloaders

```python
from pti_ldm_vae.data import create_vae_dataloaders

# Basic usage
train_loader, val_loader, _, _ = create_vae_dataloaders(
    data_base_dir="/path/to/data",
    batch_size=8,
    patch_size=(256, 256),
    data_source="edente",  # or "dente" or "both"
    rank=0
)

# With optional parameters
train_loader, val_loader, _, _ = create_vae_dataloaders(
    data_base_dir="/path/to/data",
    batch_size=8,
    patch_size=(256, 256),
    data_source="edente",
    train_split=0.85,      # 85/15 split
    num_workers=8,
    seed=42,
    cache_rate=0.5,        # cache 50% in RAM
    distributed=True,      # for DDP
    world_size=4,
    rank=rank
)
```

### VAE Inference/Evaluation Dataloader

```python
from pti_ldm_vae.data import create_vae_inference_dataloader

dataloader, image_paths = create_vae_inference_dataloader(
    input_dir="/path/to/test_images",
    patch_size=(256, 256),
    batch_size=8,
    num_samples=100,    # optional
    num_workers=4       # optional
)
```

## Data Directory Structure

```
data_base_dir/
├── edente/
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
└── dente/
    ├── image_001.tif
    ├── image_002.tif
    └── ...
```

Images can be stored in `edente/`, `dente/`, or both; VAE training can target a single source or mix both datasets.

______________________________________________________________________

## Parameters

### Common Parameters

| Parameter       | Type            | Default  | Description                                |
| --------------- | --------------- | -------- | ------------------------------------------ |
| `data_base_dir` | str             | required | Base directory containing image subfolders |
| `batch_size`    | int             | required | Batch size for dataloaders                 |
| `patch_size`    | tuple[int, int] | required | Target image size (H, W)                   |
| `rank`          | int             | 0        | Process rank (for multi-GPU training)      |

### VAE-specific

| Parameter      | Type     | Default  | Description                                          |
| -------------- | -------- | -------- | ---------------------------------------------------- |
| `data_source`  | str      | "edente" | Which images: "edente", "dente", or "both"           |
| `val_dir`      | str/None | None     | Separate validation directory                        |
| `return_paths` | bool     | False    | (Implicit) Returns train/val paths alongside loaders |

### Performance Parameters

| Parameter     | Type     | Default | Description                                   |
| ------------- | -------- | ------- | --------------------------------------------- |
| `train_split` | float    | 0.9     | Train/val split ratio                         |
| `num_workers` | int      | 4       | Number of worker processes                    |
| `seed`        | int      | 42      | Random seed for reproducibility               |
| `subset_size` | int/None | None    | Use only first N images for debugging         |
| `cache_rate`  | float    | 0.0     | Fraction of dataset to cache in RAM (0.0-1.0) |
| `distributed` | bool     | False   | Use DistributedSampler for DDP                |
| `world_size`  | int      | 1       | Number of processes for DDP                   |

______________________________________________________________________

## Data Preprocessing

All images go through these transforms:

1. **Load**: Load TIF images
2. **Channel**: Ensure channel-first format [C, H, W]
3. **Resize**: Resize to `patch_size`
4. **Normalize**: Local normalization by mask (excludes background)
5. **Type**: Convert to float32 tensor

### Local Normalization

The `LocalNormalizeByMask` transform:

- Computes mean/std **only on non-zero pixels** (excludes background)
- Normalizes the entire image
- Keeps background at zero

This is important for medical images with black backgrounds.

______________________________________________________________________

## Data Split

### Standard Split

By default, data is split with configurable ratio:

- **Train**: 90% (configurable via `train_split`)
- **Val**: 10%

The split is shuffled with seed for reproducibility.

### External Validation (VAE only)

Provide a separate validation directory:

```python
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/path/to/train_data",
    val_dir="/path/to/validation_data",  # External validation
    # ... other params
)
```

When `val_dir` is provided, ALL images from `data_base_dir` are used for training.

______________________________________________________________________

## Custom Transforms

```python
from pti_ldm_vae.data import LocalNormalizeByMask, ApplyLocalNormd

# Single image transform
normalizer = LocalNormalizeByMask()
normalized_image = normalizer(image)

# Dictionary transform (for MONAI pipelines)
dict_normalizer = ApplyLocalNormd(keys=["image", "condition"])
data = dict_normalizer({"image": img1, "condition": img2})

# Full preprocessing pipeline (load -> channel-first -> resize -> normalize)
from pti_ldm_vae.data import build_vae_preprocess_transform

transform = build_vae_preprocess_transform(patch_size=(256, 256))
```

______________________________________________________________________

## Examples

### VAE Training with Different Data Sources

```python
# Train on edente images only
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=16,
    patch_size=(512, 512),
    data_source="edente"
)

# Train on both edente + dente mixed together
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=16,
    patch_size=(512, 512),
    data_source="both"  # Doubles the dataset size
)
```

### Multi-GPU Training (DDP)

```python
# VAE with DDP
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    rank=rank,
    distributed=True,
    world_size=world_size,
    seed=42
)
```

### Debug Mode

```python
# Use only 50 images for quick debugging
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=4,
    patch_size=(256, 256),
    subset_size=50,
    data_source="edente"
)
```

### Performance Optimization with Caching

```python
# Cache 50% of training data in RAM
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    cache_rate=0.5,
    num_workers=8
)
```

______________________________________________________________________

## Performance Tips

### Workers Configuration

The `num_workers` parameter controls parallel data loading:

- **Too few**: GPU waits for data (bottleneck)
- **Too many**: High RAM usage, diminishing returns
- **Recommended**: `min(4 * num_gpus, num_cpu_cores)`

### Caching for Speed

- **0.0** (default): No caching, load from disk each time
- **0.5**: Cache 50% of training data (good balance)
- **1.0**: Cache all data (fastest, requires lots of RAM)

Validation data is ALWAYS cached at 100% when caching is enabled.

### DistributedSampler for DDP

When training with multiple GPUs, enable `distributed=True`:

```python
train_loader, val_loader = create_vae_dataloaders(
    # ...
    distributed=True,
    world_size=4,  # 4 GPUs
    rank=rank
)
# Each GPU sees different data → true parallelism
```

______________________________________________________________________

## Dataset Statistics

When loading data, detailed statistics are printed (rank 0 only):

```
============================================================
Dataset Statistics
============================================================
Data source: edente
Train samples: 1080
Val samples: 120
Total: 1200
Split ratio: 90.0% / 10.0%

Image properties:
  Shape: torch.Size([1, 256, 256])
  Dtype: torch.float32
  Range: [-2.456, 3.123]
  Mean: 0.042
  Std: 0.987
============================================================
```

______________________________________________________________________

## Notes

- Train dataloaders shuffle data (with seed for reproducibility), val dataloaders don't
- Images are loaded as grayscale (1 channel)
- Background pixels (value = 0) are preserved during normalization
- All images are ALWAYS resized to `patch_size`
