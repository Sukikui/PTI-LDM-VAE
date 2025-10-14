# Models Module

This module provides simple wrappers around MONAI models. 
The wrappers simplify model instantiation from configuration dictionaries while maintaining full compatibility 
with existing checkpoints.

## Overview

- **`VAEModel`**: Wrapper around `monai.networks.nets.AutoencoderKL`
- **`DiffusionUNet`**: Wrapper around `monai.networks.nets.DiffusionModelUNet`
- **`create_condition_projector`**: Helper function to create conditioning projection layers

---

## Quick Start

```python
from pti_ldm_vae.models import VAEModel, DiffusionUNet, create_condition_projector

# Create VAE from configuration
vae_config = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "latent_channels": 4,
    "channels": [64, 128, 256],
    "num_res_blocks": 2,
    "norm_num_groups": 32,
    "attention_levels": [False, False, False],
    "with_encoder_nonlocal_attn": True,
    "with_decoder_nonlocal_attn": True,
}
vae = VAEModel.from_config(vae_config).to(device)

# Create LDM with UNet from configuration
unet_config = {
    "spatial_dims": 2,
    "in_channels": 4,
    "out_channels": 4,
    "channels": [32, 64, 128, 256],
    "attention_levels": [False, True, True, True],
    "num_head_channels": [0, 32, 32, 32],
    "num_res_blocks": 2,
    "with_conditioning": True,
    "cross_attention_dim": 512,
}
unet = DiffusionUNet.from_config(unet_config).to(device)

# Create condition projector (for diffusion training)
condition_projector = create_condition_projector(
    condition_input_dim=4,  # VAE latent channels
    cross_attention_dim=512  # UNet cross-attention dimension
).to(device)
```

---

## VAEModel

### Basic Usage

```python
# Training
reconstruction, z_mu, z_sigma = vae(images)

# Encoding for diffusion (stage 2)
latent = vae.encode_stage_2_inputs(images)

# Decoding from diffusion output
decoded = vae.decode_stage_2_outputs(latent)
```

### Loading Pretrained Weights

```python
# Load checkpoint
checkpoint = torch.load("path/to/autoencoder.pt", map_location=device)
vae.load_state_dict(checkpoint)

# For DDP models, the checkpoint structure is automatically handled
if ddp_bool:
    vae = DDP(vae, device_ids=[device])
```

### Saving Weights

```python
# Save model
torch.save(vae.state_dict(), "autoencoder.pt")

# For DDP models
if ddp_bool:
    torch.save(vae.module.state_dict(), "autoencoder.pt")
```

---

## DiffusionUNet

### Basic Usage

```python
# Forward pass with conditioning
noise_pred = unet(z_noisy, timesteps=t, context=condition_context)

# Forward pass without conditioning
noise_pred = unet(z_noisy, timesteps=t)
```

### With Conditioning

The typical workflow for conditioned diffusion training:

```python
# 1. Encode condition images to latent space
condition_latent = vae.encode_stage_2_inputs(condition_images)  # [B, 4, H, W]

# 2. Reshape latent to sequence format for cross-attention
B, C, H, W = condition_latent.shape
condition_seq = condition_latent.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, 4]

# 3. Project to cross-attention dimension
condition_context = condition_projector(condition_seq)  # [B, N, 512]

# 4. Use in diffusion forward pass
noise_pred = unet(z_noisy, timesteps=t, context=condition_context)
```

### Loading and Saving Checkpoints

```python
# Load UNet weights
unet.load_state_dict(torch.load("diffusion_unet.pt", map_location=device))

# Save complete checkpoint (including projector)
torch.save({
    'epoch': epoch,
    'unet_state_dict': unet.state_dict(),
    'condition_projector_state_dict': condition_projector.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_loss': best_val_loss,
}, "checkpoint.pth")

# Load complete checkpoint
checkpoint = torch.load("checkpoint.pth", map_location=device)
unet.load_state_dict(checkpoint['unet_state_dict'])
condition_projector.load_state_dict(checkpoint['condition_projector_state_dict'])
```

---

## Condition Projector

The condition projector transforms condition features (e.g., VAE latents) to the cross-attention dimension
expected by the UNet.

```python
# Create projector
projector = create_condition_projector(
    condition_input_dim=4,      # Input: VAE latent channels
    cross_attention_dim=512     # Output: UNet cross-attention dim
)

# Use in optimizer (train both UNet and projector together)
optimizer = torch.optim.Adam(
    list(unet.parameters()) + list(projector.parameters()),
    lr=1e-5
)
```

---

## Loss Functions

### KL Divergence Loss

The `compute_kl_loss` function calculates the KL divergence between the learned latent distribution and a standard Gaussian prior.

```python
from pti_ldm_vae.models import compute_kl_loss

# During VAE training
reconstruction, z_mu, z_sigma = vae(images)
kl_loss = compute_kl_loss(z_mu, z_sigma)

# Combine with reconstruction loss
recons_loss = intensity_loss(reconstruction, images)
loss = recons_loss + kl_weight * kl_loss
```

**Parameters:**
- `z_mu` (torch.Tensor): Mean of the latent distribution, shape `[B, C, ...]`
- `z_sigma` (torch.Tensor): Standard deviation of the latent distribution, shape `[B, C, ...]`

**Returns:**
- Scalar KL divergence loss averaged over the batch

---

## Configuration Reference

### VAE Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spatial_dims` | int | required | 2 for 2D, 3 for 3D |
| `in_channels` | int | required | Number of input channels |
| `out_channels` | int | required | Number of output channels |
| `latent_channels` | int | required | Latent space channels |
| `channels` | list[int] | required | Channels per resolution level |
| `num_res_blocks` | int | 2 | Residual blocks per level |
| `norm_num_groups` | int | 32 | Group norm groups |
| `norm_eps` | float | 1e-6 | Normalization epsilon |
| `attention_levels` | list[bool] | `[False]*len(channels)` | Enable attention per level |
| `with_encoder_nonlocal_attn` | bool | True | Non-local attention in encoder |
| `with_decoder_nonlocal_attn` | bool | True | Non-local attention in decoder |

### UNet Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spatial_dims` | int | required | 2 for 2D, 3 for 3D |
| `in_channels` | int | required | Input channels (VAE latent channels) |
| `out_channels` | int | required | Output channels (usually same as input) |
| `channels` | list[int] | required | Channels per resolution level |
| `attention_levels` | list[bool] | required | Enable attention per level |
| `num_head_channels` | list[int] | required | Attention heads per level |
| `num_res_blocks` | int | 2 | Residual blocks per level |
| `with_conditioning` | bool | True | Enable cross-attention conditioning |
| `cross_attention_dim` | int | 512 | Cross-attention feature dimension |
| `norm_num_groups` | int | 32 | Group norm groups |


---

## DDP (Distributed Data Parallel) Usage

```python
# Wrap models with DDP
if ddp_bool:
    vae = DDP(vae, device_ids=[device], output_device=rank, find_unused_parameters=True)
    unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

# Save with DDP (unwrap module)
if ddp_bool:
    torch.save(vae.module.state_dict(), "vae.pt")
    torch.save(unet.module.state_dict(), "unet.pt")
else:
    torch.save(vae.state_dict(), "vae.pt")
    torch.save(unet.state_dict(), "unet.pt")
```
