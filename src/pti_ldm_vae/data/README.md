# Data Module

This module handles data loading and preprocessing for TIF images.

## Overview

- **`create_vae_dataloaders`**: Load single images for VAE training
- **`create_ldm_dataloaders`**: Load paired images for LDM training
- **Transform classes**: Custom preprocessing (normalization, masking)

---

## Quick Start

### VAE Dataloaders

Load images for VAE training. You can choose to load:
- Only "edente" images (default)
- Only "dente" images
- Both mixed together

```python
from pti_ldm_vae.data import create_vae_dataloaders

# Load only edente images (default)
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/path/to/data",
    batch_size=8,
    patch_size=(256, 256),
    data_source="edente"  # or "dente" or "both"
)

# Iterate over batches
for images in train_loader:
    # images shape: [B, 1, H, W]
    pass
```

### LDM Dataloaders

Load paired images from "edente" (target) and "dente" (condition) folders:

```python
from pti_ldm_vae.data import create_ldm_dataloaders

train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/path/to/data",
    batch_size=8,
    patch_size=(256, 256),
    augment=False,
    rank=0
)

# Iterate over batches
for images, condition_images in train_loader:
    # images shape: [B, 1, H, W]
    # condition_images shape: [B, 1, H, W]
    pass
```

---

## Data Directory Structure

Your data should be organized as follows:

```
data_base_dir/
├── edente/
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
└── dente/              # Only for LDM
    ├── image_001.tif
    ├── image_002.tif
    └── ...
```

**Important**: For LDM, the "edente" and "dente" folders must contain the **same number** of images with **matching names**.

---

## Parameters

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_base_dir` | str | required | Base directory containing "edente" (and "dente" for LDM) |
| `batch_size` | int | required | Batch size for dataloaders |
| `patch_size` | tuple[int, int] | required | Target image size (H, W) |
| `augment` | bool | False | Enable data augmentation |
| `rank` | int | 0 | Process rank (for multi-GPU training) |
| `data_source` | str | "edente" | VAE only: "edente", "dente", or "both" |

### Augmentation

When `augment=True`:
- **VAE**: Applies augmentation + resizes to `patch_size`
- **LDM**: Applies augmentation to both images + resizes to `patch_size`

Requires `augmentation_utils.py` with `get_albumentations_transform()` function.

---

## Data Preprocessing

All images go through these transforms:

1. **Load**: Load TIF images
2. **Channel**: Ensure channel-first format [C, H, W]
3. **Resize**: Resize to `patch_size` (if augmentation enabled)
4. **Normalize**: Local normalization by mask (excludes background)
5. **Augmentation**: Apply augmentations (if enabled)
6. **Type**: Convert to float32 tensor

### Local Normalization

The `LocalNormalizeByMask` transform:
- Computes mean/std **only on non-zero pixels** (excludes background)
- Normalizes the entire image
- Keeps background at zero

This is important for medical images with black backgrounds.

---

## Data Split

Both functions automatically split data into train/val sets:
- **Train**: 90% of data
- **Val**: 10% of data

The split is deterministic (based on sorted file names).

---

## Custom Transforms

If you need to use the transforms directly:

```python
from pti_ldm_vae.data import LocalNormalizeByMask, ApplyLocalNormd, ToTuple

# Single image transform
normalizer = LocalNormalizeByMask()
normalized_image = normalizer(image)

# Dictionary transform (for MONAI pipelines)
dict_normalizer = ApplyLocalNormd(keys=["image", "condition"])
data = dict_normalizer({"image": img1, "condition": img2})

# Convert dict to tuple (for LDM dataloaders)
to_tuple = ToTuple(keys=["image", "condition"])
image_tuple = to_tuple({"image": img1, "condition": img2})
# Returns: (img1, img2)
```

---

## Examples

### Example 1: VAE Training (different data sources)

```python
from pti_ldm_vae.data import create_vae_dataloaders

# Train on edente images only
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=16,
    patch_size=(512, 512),
    data_source="edente"
)

# Train on dente images only
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=16,
    patch_size=(512, 512),
    data_source="dente"
)

# Train on both edente + dente mixed together
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=16,
    patch_size=(512, 512),
    data_source="both"  # Doubles the dataset size
)
```

### Example 2: LDM Training

```python
from pti_ldm_vae.data import create_ldm_dataloaders

train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    augment=False
)

# First batch
images, conditions = next(iter(train_loader))
print(f"Images shape: {images.shape}")
print(f"Conditions shape: {conditions.shape}")
```

### Example 3: DDP Multi-GPU Training

```python
from pti_ldm_vae.data import create_vae_dataloaders

# rank and world_size come from DDP setup
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    rank=rank  # Only rank 0 will print info
)
```

---

## Notes

- All dataloaders use **4 workers** and **pin_memory=True** for performance
- Train dataloaders shuffle data, val dataloaders don't
- Images are loaded as grayscale (1 channel)
- Background pixels (value = 0) are preserved during normalization