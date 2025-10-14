# Utils Module

This module provides utility functions for distributed training and visualization.

## Overview

- **`setup_ddp`**: Setup distributed data parallel training
- **`normalize_batch_for_display`**: Normalize image batches for TensorBoard visualization
- **`normalize_image_to_uint8`**: Normalize image to uint8 format
- **`visualize_2d_image`**: Prepare 2D image for visualization as RGB
- **`visualize_one_slice_in_3d_image`**: Extract and visualize a slice from 3D image

---

## Quick Start

### Distributed Training Setup

Setup distributed training with PyTorch DDP:

```python
from pti_ldm_vae.utils.distributed import setup_ddp

# In your training script
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

dist, device = setup_ddp(rank, world_size)
```

### Visualization

Normalize image batches for TensorBoard display:

```python
from pti_ldm_vae.utils.visualization import normalize_batch_for_display
import torch

# Normalize batch for display
batch = torch.randn(8, 1, 256, 256)  # [B, C, H, W]
normalized = normalize_batch_for_display(batch)

# Now ready for TensorBoard
tensorboard_writer.add_image("images", normalized[0], global_step=step)
```

---

## Distributed Training

### `setup_ddp(rank: int, world_size: int)`

Setup distributed data parallel training environment.

**Parameters:**
- `rank` (int): Rank of current process
- `world_size` (int): Total number of processes

**Returns:**
- `dist` (module): torch.distributed module
- `device` (int): CUDA device ID for this process

**Example:**

```python
import os
from pti_ldm_vae.utils.distributed import setup_ddp

# Get rank and world_size from environment
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

# Setup DDP
dist, device = setup_ddp(rank, world_size)

# Use device for model
model = MyModel().to(device)

# Wrap with DDP
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[device], output_device=rank)
```

---

## Visualization

### `normalize_batch_for_display(tensor: torch.Tensor, low: int = 2, high: int = 98)`

Normalize image batch for visualization using percentile-based normalization.

This function normalizes each image in the batch independently, using only non-zero pixels
(to exclude background). It's designed for medical images with black backgrounds.

**Parameters:**
- `tensor` (torch.Tensor): Input batch of shape `[B, C, H, W]`
- `low` (int): Lower percentile for normalization (default: 2)
- `high` (int): Upper percentile for normalization (default: 98)

**Returns:**
- `torch.Tensor`: Normalized batch in range [0, 1], same shape as input

**Example:**

```python
from pti_ldm_vae.utils.visualization import normalize_batch_for_display
import torch
from torch.utils.tensorboard import SummaryWriter

# Create batch of images
batch = torch.randn(8, 1, 256, 256)  # [B, C, H, W]

# Normalize for display
normalized = normalize_batch_for_display(batch)

# Log to TensorBoard
writer = SummaryWriter("runs/experiment")
writer.add_image("batch/image_0", normalized[0], global_step=0)

# Or create a grid
from torchvision.utils import make_grid
grid = make_grid(normalized, nrow=4)
writer.add_image("batch/grid", grid, global_step=0)
```

### How it works

1. **Per-image normalization**: Each image in the batch is normalized independently
2. **Mask-based**: Only non-zero pixels are used to compute statistics (excludes background)
3. **Percentile clipping**: Uses 2nd and 98th percentiles to avoid outliers
4. **Output range**: Final values are clipped to [0, 1]

### Notes

- Background pixels (value = 0) remain at 0 after normalization
- Very small values (< 1e-3) are set to 0 for cleaner visualization
- Works with any number of channels
- Useful for visualizing medical images with varying intensity ranges

---

## Complete Training Example

```python
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from pti_ldm_vae.utils.distributed import setup_ddp
from pti_ldm_vae.utils.visualization import normalize_batch_for_display

# Setup distributed training
rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist, device = setup_ddp(rank, world_size)

# Create model and wrap with DDP
model = MyModel().to(device)
model = DDP(model, device_ids=[device], output_device=rank)

# TensorBoard (only on rank 0)
if rank == 0:
    writer = SummaryWriter("runs/experiment")

# Training loop
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        images = batch.to(device)

        # Forward pass
        outputs = model(images)

        # Visualization (only on rank 0)
        if rank == 0 and step == 0:
            # Normalize for display
            img_disp = normalize_batch_for_display(images[:4])
            out_disp = normalize_batch_for_display(outputs[:4])

            # Log to TensorBoard
            writer.add_image("train/input", img_disp[0], global_step=epoch)
            writer.add_image("train/output", out_disp[0], global_step=epoch)
```

---

## Image Visualization Functions

### `normalize_image_to_uint8(image: np.ndarray)`

Normalize a single image to uint8 format [0, 255] for saving or display.

**Parameters:**
- `image` (np.ndarray): Input image as numpy array

**Returns:**
- `np.ndarray`: Image normalized to uint8 format

**Example:**

```python
from pti_ldm_vae.utils.visualization import normalize_image_to_uint8
import numpy as np

image = np.random.randn(256, 256)
normalized = normalize_image_to_uint8(image)
# normalized is now in range [0, 255] as uint8
```

---

### `visualize_2d_image(image)`

Prepare a 2D grayscale image for visualization by converting it to RGB format.

**Parameters:**
- `image`: Image array of shape (H, W), can be torch tensor or numpy array

**Returns:**
- `np.ndarray`: RGB image of shape (H, W, 3) with values in [0, 255]

**Example:**

```python
from pti_ldm_vae.utils.visualization import visualize_2d_image
import torch
from PIL import Image

# From torch tensor
image_tensor = torch.randn(256, 256)
rgb_image = visualize_2d_image(image_tensor)

# Save as image
Image.fromarray(rgb_image).save("output.png")
```

---

### `visualize_one_slice_in_3d_image(image, axis: int = 2)`

Extract the center slice from a 3D image and prepare it for visualization as RGB.

**Parameters:**
- `image`: 3D image array of shape (H, W, D), can be torch tensor or numpy array
- `axis` (int): Axis along which to take the center slice (0, 1, or 2). Default: 2

**Returns:**
- `np.ndarray`: RGB image of shape (H, W, 3) with values in [0, 255]

**Raises:**
- `ValueError`: If axis is not in [0, 1, 2]

**Example:**

```python
from pti_ldm_vae.utils.visualization import visualize_one_slice_in_3d_image
import torch
from PIL import Image

# 3D image
image_3d = torch.randn(128, 128, 128)

# Extract center slice along z-axis (axis=2)
rgb_slice = visualize_one_slice_in_3d_image(image_3d, axis=2)

# Save
Image.fromarray(rgb_slice).save("slice.png")
```

---

## Notes

- The `setup_ddp` function initializes NCCL backend for multi-GPU training
- The `normalize_batch_for_display` function is specifically designed for medical images with background masking
- The `visualize_2d_image` and `visualize_one_slice_in_3d_image` functions convert grayscale to RGB by stacking channels
- All visualization functions normalize images to [0, 255] uint8 range for display/saving
- All functions are used throughout the training and inference scripts