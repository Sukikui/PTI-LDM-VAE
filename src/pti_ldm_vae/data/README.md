# Data Module

This module handles data loading and preprocessing for TIF images.

## Overview

- **`create_vae_dataloaders`**: Load single images for VAE training
- **`create_ldm_dataloaders`**: Load paired images for LDM training
- **Transform classes**: Custom preprocessing (normalization, masking)

______________________________________________________________________

## Quick Start

### VAE Dataloaders

```python
from pti_ldm_vae.data import create_vae_dataloaders

# Basic usage
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/path/to/data",
    batch_size=8,
    patch_size=(256, 256),
    data_source="edente",  # or "dente" or "both"
    rank=0
)

# With optional parameters
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/path/to/data",
    batch_size=8,
    patch_size=(256, 256),
    data_source="edente",
    train_split=0.85,      # 85/15 split
    num_workers=8,         # 8 workers
    seed=42,               # reproducibility
    cache_rate=0.5,        # cache 50% in RAM
    distributed=True,      # for DDP
    world_size=4,
    rank=rank
)
```

### LDM Dataloaders

Load paired images from target and condition folders:

```python
from pti_ldm_vae.data import create_ldm_dataloaders

# Basic usage
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/path/to/data",
    batch_size=8,
    patch_size=(256, 256),
    target="edente",      # target folder
    condition="dente",    # condition folder
    rank=0
)

# With optional parameters
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/path/to/data",
    batch_size=8,
    patch_size=(256, 256),
    target="edente",
    condition="dente",
    train_split=0.85,      # 85/15 split
    num_workers=8,         # 8 workers
    seed=42,               # reproducibility
    cache_rate=0.5,        # cache 50% in RAM
    distributed=True,      # for DDP
    world_size=4,
    rank=rank
)

# Iterate over batches
for target_images, condition_images in train_loader:
    # target_images shape: [B, 1, H, W]
    # condition_images shape: [B, 1, H, W]
    pass
```

______________________________________________________________________

## Data Directory Structure

Your data should be organized as follows:

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

**Important**: For LDM, the "edente" and "dente" folders must contain the **same number** of images with **matching names**.

______________________________________________________________________

## Parameters

### Common Parameters

| Parameter       | Type            | Default  | Description                                |
| --------------- | --------------- | -------- | ------------------------------------------ |
| `data_base_dir` | str             | required | Base directory containing image subfolders |
| `batch_size`    | int             | required | Batch size for dataloaders                 |
| `patch_size`    | tuple[int, int] | required | Target image size (H, W)                   |
| `augment`       | bool            | False    | Enable data augmentation                   |
| `rank`          | int             | 0        | Process rank (for multi-GPU training)      |

### VAE-specific Parameters

| Parameter     | Type        | Default  | Description                                           |
| ------------- | ----------- | -------- | ----------------------------------------------------- |
| `data_source` | str         | "edente" | Which images to load: "edente", "dente", or "both"    |
| `val_dir`     | str \| None | None     | Separate validation directory (overrides train_split) |

### LDM-specific Parameters

| Parameter   | Type | Default  | Description            |
| ----------- | ---- | -------- | ---------------------- |
| `target`    | str  | "edente" | Target image folder    |
| `condition` | str  | "dente"  | Condition image folder |

### Shared Performance Parameters (VAE + LDM)

| Parameter     | Type        | Default | Description                                              |
| ------------- | ----------- | ------- | -------------------------------------------------------- |
| `train_split` | float       | 0.9     | Train/val split ratio (e.g., 0.9 = 90% train, 10% val)   |
| `num_workers` | int         | 4       | Number of worker processes for data loading              |
| `seed`        | int \| None | 42      | Random seed for reproducibility (None = no seed)         |
| `subset_size` | int \| None | None    | Use only first N images/pairs for debugging (None = all) |
| `cache_rate`  | float       | 0.0     | Fraction of dataset to cache in RAM (0.0 to 1.0)         |
| `distributed` | bool        | False   | Use DistributedSampler for DDP training                  |
| `world_size`  | int         | 1       | Number of processes for DDP                              |

### Augmentation

When `augment=True`:

- **VAE**: Applies augmentation (images already resized to `patch_size`)
- **LDM**: Applies augmentation to both images (images already resized to `patch_size`)

**Note**: Images are ALWAYS resized to `patch_size` regardless of augmentation setting.

Augmentations are defined in `augmentation.py` using albumentations library.

______________________________________________________________________

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

______________________________________________________________________

## Data Split

### Standard Split (VAE + LDM)

By default, data is split into train/val sets with configurable ratio:

- **Train**: 90% of data (configurable via `train_split` parameter)
- **Val**: 10% of data

The split is **shuffled with seed** for reproducibility.

### External Validation (VAE only)

For VAE, you can provide a separate validation directory via `val_dir` parameter:

```python
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/path/to/train_data",
    val_dir="/path/to/validation_data",  # External validation
    # ... other params
)
```

When `val_dir` is provided, ALL images from `data_base_dir` are used for training.

Note: LDM does not support external validation directory yet.

______________________________________________________________________

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

______________________________________________________________________

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

### Example 2: LDM Training (Basic)

```python
from pti_ldm_vae.data import create_ldm_dataloaders

# Train edente→dente (default)
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    target="edente",
    condition="dente"
)

# Train dente→edente (reverse)
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    target="dente",
    condition="edente"
)

# First batch
target_imgs, condition_imgs = next(iter(train_loader))
print(f"Target shape: {target_imgs.shape}")
print(f"Condition shape: {condition_imgs.shape}")
```

### Example 3: DDP Multi-GPU Training (VAE + LDM)

```python
from pti_ldm_vae.data import create_vae_dataloaders, create_ldm_dataloaders

# VAE with DDP
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    rank=rank,
    distributed=True,  # Enable DistributedSampler
    world_size=world_size,  # Number of GPUs
    seed=42  # For reproducibility
)

# LDM with DDP
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    target="edente",
    condition="dente",
    rank=rank,
    distributed=True,
    world_size=world_size,
    seed=42
)
```

### Example 4: Debug Mode (VAE + LDM)

```python
from pti_ldm_vae.data import create_vae_dataloaders, create_ldm_dataloaders

# VAE: Use only 50 images for quick debugging
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=4,
    patch_size=(256, 256),
    subset_size=50,  # Only first 50 images
    data_source="edente"
)

# LDM: Use only 50 pairs for quick debugging
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=4,
    patch_size=(256, 256),
    subset_size=50,  # Only first 50 pairs
    target="edente",
    condition="dente"
)
```

### Example 5: Performance Optimization with Caching (VAE + LDM)

```python
from pti_ldm_vae.data import create_vae_dataloaders, create_ldm_dataloaders

# VAE: Cache 50% of training data in RAM for faster loading
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    cache_rate=0.5,  # Cache 50% of training data
    num_workers=8  # Use 8 workers for data loading
)

# LDM: Same caching support
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    target="edente",
    condition="dente",
    cache_rate=0.5,
    num_workers=8
)
```

### Example 6: External Validation Set

```python
from pti_ldm_vae.data import create_vae_dataloaders

# Use separate directory for validation
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/train",
    val_dir="/data/validation",  # External validation
    batch_size=8,
    patch_size=(256, 256),
    data_source="edente"
)
```

### Example 7: Custom Split Ratio (VAE + LDM)

```python
from pti_ldm_vae.data import create_vae_dataloaders, create_ldm_dataloaders

# VAE: Use 85/15 train/val split
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    train_split=0.85,  # 85% train, 15% val
    data_source="edente"
)

# LDM: Same configurable split
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data/my_project",
    batch_size=8,
    patch_size=(256, 256),
    target="edente",
    condition="dente",
    train_split=0.85  # 85% train, 15% val
)
```

______________________________________________________________________

## Performance Tips

### Workers Configuration

The `num_workers` parameter controls parallel data loading:

- **Too few workers**: GPU waits for data (bottleneck)
- **Too many workers**: High RAM usage, diminishing returns
- **Recommended**: `min(4 * num_gpus, num_cpu_cores)`

Example:

```python
# For 2 GPUs and 16 CPU cores
train_loader, val_loader = create_vae_dataloaders(
    # ...
    num_workers=8  # 4 * 2 GPUs
)
```

### Caching for Speed

Use `cache_rate` to cache dataset in RAM:

- **0.0** (default): No caching, load from disk each time
- **0.5**: Cache 50% of training data (good balance)
- **1.0**: Cache all data (fastest, requires lots of RAM)

Validation data is ALWAYS cached at 100% when caching is enabled.

### DistributedSampler for DDP

When training with multiple GPUs, enable `distributed=True`:

```python
# Without DistributedSampler (BAD)
# All GPUs see the same data → no real parallelism

# With DistributedSampler (GOOD)
train_loader, val_loader = create_vae_dataloaders(
    # ...
    distributed=True,
    world_size=4,  # 4 GPUs
    rank=rank
)
# Each GPU sees different data → true parallelism
```

## Dataset Statistics

When loading data, detailed statistics are printed automatically (rank 0 only):

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

## Configuration pour cluster

### Paramètres clés

| Paramètre     | Quoi ?                                 | Recommandation                          |
| ------------- | -------------------------------------- | --------------------------------------- |
| `num_workers` | Processus CPU pour charger les données | `4 * nb_GPUs` ou `nb_CPU_cores`         |
| `world_size`  | Nombre de GPUs pour DDP                | Nombre de GPUs disponibles (1, 2, 4, 8) |
| `rank`        | ID du GPU actuel (0 à world_size - 1)  | Automatique avec torchrun               |
| `cache_rate`  | Fraction du dataset en RAM (0.0 à 1.0) | 0.5 à 1.0 si >64GB RAM                  |
| `distributed` | Active DistributedSampler pour DDP     | `True` si world_size > 1                |

### Différence `num_workers` vs `world_size`

- **`num_workers`** : Processus **CPU** qui chargent les données en parallèle (évite que GPU attende)
- **`world_size`** : Nombre de **GPUs** qui entraînent en parallèle (accélère l'entraînement)

**Exemple :** Machine avec 4 GPUs et 16 CPU cores

- `world_size=4` → 4 GPUs entraînent en parallèle
- `num_workers=8` → Chaque GPU a 8 workers CPU → **32 workers CPU au total**

### Configuration optimale selon votre machine

#### Machine locale (1 GPU, 8 cores, 16GB RAM)

```python
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data",
    batch_size=4,
    patch_size=(256, 256),
    num_workers=4,       # 4 workers CPU
    cache_rate=0.0,      # Pas assez de RAM
    distributed=False,   # 1 seul GPU
    rank=0
)
```

#### Serveur moyen (4 GPUs, 32 cores, 128GB RAM)

```python
# VAE
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data",
    batch_size=8,
    patch_size=(256, 256),
    num_workers=8,       # 8 workers par GPU
    cache_rate=0.5,      # Cache 50% en RAM
    distributed=True,    # Multi-GPU
    world_size=4,        # 4 GPUs
    rank=rank            # Auto avec torchrun
)

# LDM (mêmes paramètres)
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data",
    batch_size=8,
    patch_size=(256, 256),
    target="edente",
    condition="dente",
    num_workers=8,
    cache_rate=0.5,
    distributed=True,
    world_size=4,
    rank=rank
)

# Lancer avec torchrun :
# torchrun --nproc_per_node=4 scripts/train_vae.py -e config/env.json -c config/config.json
```

#### Cluster HPC (8 GPUs, 128 cores, 512GB RAM)

```python
# VAE
train_loader, val_loader = create_vae_dataloaders(
    data_base_dir="/data",
    batch_size=16,
    patch_size=(256, 256),
    num_workers=16,      # 16 workers par GPU
    cache_rate=1.0,      # Cache tout en RAM
    distributed=True,    # Multi-GPU
    world_size=8,        # 8 GPUs
    rank=rank            # Auto avec torchrun
)

# LDM (mêmes paramètres)
train_loader, val_loader = create_ldm_dataloaders(
    data_base_dir="/data",
    batch_size=16,
    patch_size=(256, 256),
    target="edente",
    condition="dente",
    num_workers=16,
    cache_rate=1.0,
    distributed=True,
    world_size=8,
    rank=rank
)

# Lancer avec torchrun :
# torchrun --nproc_per_node=8 scripts/train_vae.py -e config/env.json -c config/config.json
```

### Recommandations générales

**RAM disponible :**

- < 32GB : `cache_rate=0.0` (pas de cache)
- 32-128GB : `cache_rate=0.3-0.5` (cache partiel)
- > 128GB : `cache_rate=0.8-1.0` (cache total)

**CPU cores :**

- `num_workers = min(4 * nb_GPUs, nb_CPU_cores // 2)`
- Laissez quelques cores libres pour le système

**Batch size :**

- Plus de GPUs → Augmenter batch_size proportionnellement
- 1 GPU : batch_size = 4-8
- 4 GPUs : batch_size = 8-16
- 8 GPUs : batch_size = 16-32

______________________________________________________________________

## Notes

- Train dataloaders shuffle data (with seed for reproducibility), val dataloaders don't
- Images are loaded as grayscale (1 channel)
- Background pixels (value = 0) are preserved during normalization
- All images are ALWAYS resized to `patch_size`
