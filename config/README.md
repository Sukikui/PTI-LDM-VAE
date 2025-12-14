# VAE Configuration Parameters Explained

This document explains the parameters in VAE config files (e.g., `ar_vae_edente.json`, `ar_vae_dente.json`, `ar_vae_both.json`).

______________________________________________________________________

## Paths

| Parameter        | Type    | Description                                                    |
| ---------------- | ------- | -------------------------------------------------------------- |
| `data_base_dir`  | string  | Base directory containing image folders (edente/, dente/)      |
| `run_dir`        | string  | Output directory for checkpoints, logs, and validation samples |
| `resume_ckpt`    | boolean | Set to `true` to resume training from checkpoint               |
| `checkpoint_dir` | string  | Path to checkpoint file (only used if `resume_ckpt=true`)      |

______________________________________________________________________

## Data Settings

| Parameter      | Type        | Default  | Description                                                    |
| -------------- | ----------- | -------- | -------------------------------------------------------------- |
| `data_source`  | string      | "edente" | Which images to load: **"edente"**, **"dente"**, or **"both"** |
| `train_split`  | float       | 0.9      | Train/val split ratio (0.9 = 90% train, 10% val)               |
| `val_dir`      | string/null | null     | Separate validation directory (overrides `train_split` if set) |
| `augment`      | boolean     | false    | Enable data augmentation                                       |
| `spatial_dims` | int         | 2        | Image dimensionality: **2** for 2D, **3** for 3D               |

______________________________________________________________________

## Model Architecture (`autoencoder_def`)

### Basic Structure

| Parameter      | Type | Default | Description                             |
| -------------- | ---- | ------- | --------------------------------------- |
| `spatial_dims` | int  | 2       | Must match top-level `spatial_dims`     |
| `in_channels`  | int  | 1       | Input image channels (1 for grayscale)  |
| `out_channels` | int  | 1       | Output image channels (1 for grayscale) |

### Latent Space

| Parameter         | Type | Default | Description                        | Impact                                                                                                      |
| ----------------- | ---- | ------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `latent_channels` | int  | 4       | Number of channels in latent space | **Higher** = more information preserved, less compression<br>**Lower** = more compression, may lose details |

**Typical values:**

- `4`: Standard compression (recommended)
- `8`: Less compression, better quality
- `2`: High compression, may be too lossy

### Model Capacity

| Parameter  | Type      | Default        | Description                                        | Impact                                                                                                                         |
| ---------- | --------- | -------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `channels` | list[int] | [64, 128, 256] | Channel progression through encoder/decoder levels | **Higher values** = more parameters, better quality, slower training<br>**More levels** = deeper model, larger receptive field |

**Typical configurations:**

- **Small**: `[32, 64, 128]` - Faster, less memory
- **Standard**: `[64, 128, 256]` - Good balance (recommended)
- **Large**: `[128, 256, 512]` - Better quality, more memory

| Parameter        | Type | Default | Description                         | Impact                                               |
| ---------------- | ---- | ------- | ----------------------------------- | ---------------------------------------------------- |
| `num_res_blocks` | int  | 2       | Number of residual blocks per level | **More blocks** = better quality but slower training |

**Typical values:**

- `1`: Fast training, lower quality
- `2`: Good balance (recommended)
- `3-4`: Better quality, slower

### Normalization

| Parameter         | Type  | Default | Description                     | Constraints                                                                     |
| ----------------- | ----- | ------- | ------------------------------- | ------------------------------------------------------------------------------- |
| `norm_num_groups` | int   | 32      | Number of groups for GroupNorm  | **Must divide all channel counts evenly**<br>Example: 32 divides 64, 128, 256 ✓ |
| `norm_eps`        | float | 1e-06   | Epsilon for numerical stability | Rarely needs changing                                                           |

**Common values for `norm_num_groups`:**

- If `channels = [64, 128, 256]`: Use **32** (divides all)
- If `channels = [48, 96, 192]`: Use **24** or **48**
- If `channels = [128, 256, 512]`: Use **32** or **64**

### Attention Mechanisms

| Parameter          | Type       | Default               | Description                                                     | Impact                                                     |
| ------------------ | ---------- | --------------------- | --------------------------------------------------------------- | ---------------------------------------------------------- |
| `attention_levels` | list[bool] | [false, false, false] | Enable self-attention at each level<br>[level1, level2, level3] | **true** = slower but better quality<br>**false** = faster |

**Recommendations:**

- For 256×256 images: `[false, false, false]` (faster, usually sufficient)
- For high-quality results: `[false, true, true]` (attention at deeper levels)
- For maximum quality: `[true, true, true]` (very slow)

| Parameter                    | Type    | Default | Description                               | Impact                                                                |
| ---------------------------- | ------- | ------- | ----------------------------------------- | --------------------------------------------------------------------- |
| `with_encoder_nonlocal_attn` | boolean | true    | Non-local attention in encoder bottleneck | Captures **global context** across the image<br>Recommended: **true** |
| `with_decoder_nonlocal_attn` | boolean | true    | Non-local attention in decoder bottleneck | Improves **reconstruction quality**<br>Recommended: **true**          |

______________________________________________________________________

## Training Hyperparameters (`autoencoder_train`)

### Basic Training Settings

| Parameter      | Type      | Default    | Description                      |
| -------------- | --------- | ---------- | -------------------------------- |
| `batch_size`   | int       | 8          | Batch size per GPU               |
| `patch_size`   | list[int] | [256, 256] | Image patch size [Height, Width] |
| `max_epochs`   | int       | 100        | Maximum training epochs          |
| `val_interval` | int       | 1          | Run validation every N epochs    |

**Batch size recommendations:**

- Single GPU: 4-8
- 4 GPUs: 16-32

### Learning Rate

| Parameter | Type  | Default | Description   | Important Notes                                                                                                         |
| --------- | ----- | ------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `lr`      | float | 2.5e-5  | Learning rate | **Auto-scaled** by `world_size` in multi-GPU training<br>Single GPU: 2.5e-5<br>4 GPUs: Effective LR = 2.5e-5 × 4 = 1e-4 |

**Typical values:**

- `1e-5` to `5e-5`: Safe range
- Too high → Training instability
- Too low → Slow convergence

### Loss Weights

| Parameter           | Type  | Default | Description                    | Impact                                                                                                                      |
| ------------------- | ----- | ------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `perceptual_weight` | float | 1.0     | Weight for VGG perceptual loss | **Higher** = better texture quality<br>**Lower** = faster convergence<br>**0.0** = disable perceptual loss                  |
| `adv_weight`        | float | 0.5     | Weight for adversarial losses  | Scales both generator and discriminator adversarial losses. Increase for stronger GAN signal, decrease if GAN destabilizes. |

**Typical range:** 0.5 to 2.0

| Parameter   | Type  | Default | Description                   | ⚠️ Critical Trade-off                                                                                                   |
| ----------- | ----- | ------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `kl_weight` | float | 1e-6    | Weight for KL divergence loss | **Too high** → Blurry reconstructions<br>**Too low** → Unstructured latent space<br>**Recommended range:** 1e-6 to 1e-5 |

**How to tune `kl_weight`:**

1. **Reconstructions are blurry?** → Decrease `kl_weight` (e.g., 1e-6 → 5e-7)
2. **Latent space is chaotic?** → Increase `kl_weight` (e.g., 1e-6 → 5e-6)
3. **Start with:** `1e-6` (good default)

### Reconstruction Loss

| Parameter    | Type   | Default | Description              | Characteristics                                                                                               |
| ------------ | ------ | ------- | ------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `recon_loss` | string | "l1"    | Reconstruction loss type | **"l1"** = Sharper details, better for medical images<br>**"l2"** = Smoother results, more robust to outliers |

**Choose:**

- **"l1"**: For sharp, detailed images (recommended for medical imaging)
- **"l2"**: For smoother reconstructions

______________________________________________________________________

## Common Configuration Scenarios

### High Quality (Slower Training)

```json
{
  "autoencoder_def": {
    "latent_channels": 8,
    "channels": [128, 256, 512],
    "num_res_blocks": 3,
    "attention_levels": [false, true, true],
    "with_encoder_nonlocal_attn": true,
    "with_decoder_nonlocal_attn": true
  },
  "autoencoder_train": {
    "batch_size": 4,
    "lr": 1e-5,
    "perceptual_weight": 1.5,
    "kl_weight": 5e-7
  }
}
```

### Fast Training (Lower Quality)

```json
{
  "autoencoder_def": {
    "latent_channels": 4,
    "channels": [32, 64, 128],
    "num_res_blocks": 1,
    "attention_levels": [false, false, false],
    "with_encoder_nonlocal_attn": false,
    "with_decoder_nonlocal_attn": false
  },
  "autoencoder_train": {
    "batch_size": 16,
    "lr": 5e-5,
    "perceptual_weight": 0.5,
    "kl_weight": 1e-6
  }
}
```

### Balanced (Recommended Default)

```json
{
  "autoencoder_def": {
    "latent_channels": 4,
    "channels": [64, 128, 256],
    "num_res_blocks": 2,
    "attention_levels": [false, false, false],
    "with_encoder_nonlocal_attn": true,
    "with_decoder_nonlocal_attn": true
  },
  "autoencoder_train": {
    "batch_size": 8,
    "lr": 2.5e-5,
    "perceptual_weight": 1.0,
    "kl_weight": 1e-6
  }
}
```

______________________________________________________________________

## Troubleshooting Guide

### Problem: Blurry Reconstructions

**Possible causes:**

- `kl_weight` too high
- `perceptual_weight` too low
- Model too small

**Solutions:**

1. Decrease `kl_weight`: 1e-6 → 5e-7
2. Increase `perceptual_weight`: 1.0 → 1.5
3. Increase model capacity: `channels: [64, 128, 256]` → `[128, 256, 512]`

### Problem: Training Unstable / NaN Loss

**Possible causes:**

- Learning rate too high
- `kl_weight` too high

**Solutions:**

1. Decrease `lr`: 2.5e-5 → 1e-5
2. Decrease `kl_weight`: 1e-6 → 1e-7

### Problem: Out of Memory (OOM)

**Solutions:**

1. Decrease `batch_size`: 8 → 4
2. Decrease model size: `channels: [64, 128, 256]` → `[32, 64, 128]`
3. Decrease `patch_size`: [256, 256] → [128, 128]
4. Disable attention: Set all `attention_levels` to `false`

### Problem: Latent Space is Unstructured

**Cause:** `kl_weight` too low

**Solution:**

- Increase `kl_weight`: 1e-6 → 5e-6 or 1e-5

______________________________________________________________________

## Advanced Tips

### Multi-GPU Training

The learning rate is **automatically scaled** by the number of GPUs:

```
effective_lr = lr × world_size
```

Example:

- Config: `"lr": 2.5e-5`
- Single GPU: Effective LR = 2.5e-5
- 4 GPUs: Effective LR = 1e-4

**You don't need to change `lr` in the config for multi-GPU training!**

### Choosing `norm_num_groups`

The number of groups must **divide all channel counts evenly**:

```python
# Good example:
channels = [64, 128, 256]
norm_num_groups = 32  # ✓ 64/32=2, 128/32=4, 256/32=8

# Bad example:
channels = [64, 128, 256]
norm_num_groups = 48  # ✗ 64/48 is not an integer
```

**Quick rule:** Use **32** for most cases.

### Balancing KL Weight

The `kl_weight` is the **most important hyperparameter** to tune:

1. Start with default: `1e-6`
2. After a few epochs, check validation samples:
   - **Too blurry?** Decrease to `5e-7` or `1e-7`
   - **Sharp but chaotic latent?** Increase to `5e-6` or `1e-5`
3. Fine-tune incrementally

**Goal:** Balance between reconstruction quality and latent structure.
