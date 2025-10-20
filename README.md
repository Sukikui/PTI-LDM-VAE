# PTI-LDM-VAE

Two-stage pipeline for medical image generation using Variational Autoencoders (VAE) and Latent Diffusion Models (LDM).

## Overview

This project implements a two-stage generative pipeline for medical TIF images:

1. **Stage 1 - VAE**: Learn a compact latent representation of edentulous images
2. **Stage 2 - LDM**: Conditional diffusion model for dental→edentulous image translation

Both models operate on float32 single-channel TIF images (256×256).

**Original Author**: Tuong Vy PHAM (tv.pham1996@gmail.com)

______________________________________________________________________

## Features

- **VAE with adversarial training** (PatchGAN discriminator)
- **Conditional LDM** with cross-attention
- **Multi-GPU training** with Distributed Data Parallel (DDP)
- **Latent space visualization** (UMAP, t-SNE)
- **Comprehensive metrics** (PSNR, SSIM, Dice, IoU)
- **TensorBoard monitoring**
- **Automatic checkpoint management**

______________________________________________________________________

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd PTI-LDM-VAE

# Create conda environment
conda env create -f environment.yml
conda activate pti-ldm-vae

# Or use pip
pip install -r requirements.txt
```

### Train a VAE

```bash
python scripts/train_vae.py -c config/vae_config.json
```

### Train an LDM (requires trained VAE)

```bash
python scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1
```

### Run Inference

```bash
# VAE reconstruction
python scripts/inference_vae.py \
  --checkpoint runs/vae_baseline/trained_weights/autoencoder_epoch73.pth \
  --input-dir data/edente/ \
  --num-samples 20

# LDM generation
python scripts/inference_ldm.py \
  --checkpoint runs/ldm_experiment/trained_weights/checkpoint_epoch50.pth \
  --num-samples 10
```

______________________________________________________________________

## Project Structure

```
PTI-LDM-VAE/
├── scripts/                    # Training, inference, analysis scripts
│   ├── train_vae.py           # VAE training
│   ├── train_ldm.py           # LDM training
│   ├── inference_vae.py       # VAE inference
│   ├── inference_ldm.py       # LDM inference
│   ├── analyze_static.py      # Latent space visualization (static)
│   └── analyze_interactive.py # Latent space visualization (interactive)
├── src/pti_ldm_vae/           # Source code
│   ├── models/                # VAE and LDM models
│   ├── data/                  # Dataloaders and transforms
│   ├── utils/                 # Utilities (distributed, visualization)
│   └── analysis/              # Analysis tools
├── config/                    # Configuration files
│   ├── vae_config.json       # VAE unified config
│   ├── environment_tif.json  # LDM environment config
│   └── config_train_16g_cond.json  # LDM model config
└── runs/                      # Training outputs (auto-generated)
```

______________________________________________________________________

## Data Organization

```
data/
├── edente/              # Edentulous images (VAE training)
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
└── dente/               # Dental images (LDM conditioning)
    ├── image_001.tif
    ├── image_002.tif
    └── ...
```

**Requirements:**

- TIF format, float32, single channel
- 256×256 pixels
- For LDM: matching filenames between `edente/` and `dente/` folders

______________________________________________________________________

## Configuration

### VAE Training

Uses a single unified configuration file (`config/vae_config.json`):

```bash
# Basic training
python scripts/train_vae.py -c config/vae_config.json

# With overrides
python scripts/train_vae.py \
  -c config/vae_config.json \
  --batch-size 16 \
  --lr 5e-5 \
  --max-epochs 50

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train_vae.py \
  -c config/vae_config.json \
  -g 4
```

See `config/README.md` for detailed configuration guide.

### LDM Training

Uses two configuration files:

- `environment_tif.json` - Paths and VAE checkpoint
- `config_train_16g_cond.json` - Model architecture and hyperparameters

```bash
python scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1
```

______________________________________________________________________

## Documentation

- **`scripts/README.md`** - Detailed guide for all scripts
- **`config/README.md`** - Configuration system documentation
- **`MIGRATION_GUIDE.md`** - Migration from old configuration system
- **`src/pti_ldm_vae/data/README.md`** - Dataloader API reference
- **`src/pti_ldm_vae/models/README.md`** - Model architecture details

______________________________________________________________________

## Architecture

### VAE (Variational Autoencoder)

- **Base**: MONAI AutoencoderKL
- **Loss**: L1/L2 + KL divergence + Perceptual (VGG) + Adversarial
- **Discriminator**: PatchDiscriminator (3 layers)
- **Latent dim**: 4 channels
- **Training**: 5-epoch warm-up without adversarial loss

### LDM (Latent Diffusion Model)

- **Base**: UNet with cross-attention
- **Scheduler**: DDPM (1000 timesteps)
- **Conditioning**: Linear projection from dental latent to cross-attention dim
- **Training**: Mixed precision (AMP)
- **Scale factor**: Auto-computed from VAE

______________________________________________________________________

## Monitoring

All training scripts log to TensorBoard:

```bash
tensorboard --logdir runs/
```

Open http://localhost:6006 to view:

- Loss curves (train/validation)
- Image triplets (original | reconstruction | difference)
- Generated samples during training

______________________________________________________________________

## Examples

### Complete Workflow

```bash
# 1. Train VAE
python scripts/train_vae.py -c config/vae_config.json

# 2. Test VAE
python scripts/inference_vae.py \
  --checkpoint runs/vae_baseline/trained_weights/autoencoder_epoch73.pth \
  --input-dir data/edente/ \
  --num-samples 10

# 3. Visualize latent space
python scripts/analyze_static.py \
  --vae-weights runs/vae_baseline/trained_weights/autoencoder_epoch73.pth \
  --config-file config/vae_config.json \
  --folder-edente data/edente/ \
  --output-dir results/latent_viz \
  --method umap

# 4. Train LDM (update environment_tif.json with VAE path first)
python scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1

# 5. Generate images
python scripts/inference_ldm.py \
  --checkpoint runs/ldm_experiment/trained_weights/checkpoint_epoch50.pth \
  --num-samples 20
```

### Multi-GPU Training

```bash
# VAE with 4 GPUs
torchrun --nproc_per_node=4 scripts/train_vae.py \
  -c config/vae_config.json \
  -g 4

# LDM with 4 GPUs
torchrun --nproc_per_node=4 scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 4
```

______________________________________________________________________

## Key Dependencies

- Python 3.10+
- PyTorch 2.5.1+
- MONAI 1.5.1
- CUDA-compatible GPU (recommended)

See `requirements.txt` or `environment.yml` for complete list.

______________________________________________________________________

## Performance Tips

### GPU Memory

- Reduce `batch_size` if OOM errors occur
- Reduce `patch_size` (e.g., 256→128)
- Enable gradient checkpointing (if available)

### Training Speed

- Use `--cache-rate 1.0` to cache dataset in RAM
- Increase `--num-workers` (e.g., 8-16)
- Use multiple GPUs with DDP

### Hyperparameters

- **KL weight** (`kl_weight`): Controls latent space regularization
  - Too high → poor reconstruction
  - Too low → unstructured latent
  - Recommended: 1e-6 to 1e-5
- **Perceptual weight** (`perceptual_weight`): Improves texture quality
  - Recommended: 1.0
- **Learning rate** (`lr`): Auto-scaled by world_size in DDP
  - Single GPU: 2.5e-5
  - 4 GPUs: Effective LR = 2.5e-5 × 4

______________________________________________________________________

## Resuming Training

Both VAE and LDM support checkpoint resuming:

```json
{
  "resume_ckpt": true,
  "checkpoint_dir": "path/to/checkpoint_epoch73.pth"
}
```

Checkpoints include:

- Model states (autoencoder, discriminator/unet)
- Optimizer states
- Best validation loss
- Current epoch
- Global step counter

______________________________________________________________________

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size`
- Reduce `patch_size`
- Enable gradient checkpointing

### VAE produces blurry reconstructions

- Increase `perceptual_weight`
- Decrease `kl_weight`
- Increase model capacity (`channels`)

### LDM doesn't converge

- Check VAE quality first
- Verify `scale_factor` is close to 1
- Increase `num_train_timesteps`

### Validation images are black

- Check data normalization
- Verify background masking works correctly
- Adjust percentiles in `normalize_batch_for_display`

______________________________________________________________________

## License

Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0

______________________________________________________________________

## Citation

If you use this code, please cite the original MONAI framework:

```bibtex
@article{monai2020,
  title={MONAI: An open-source framework for deep learning in healthcare imaging},
  author={MONAI Consortium},
  journal={arXiv preprint arXiv:2211.02701},
  year={2022}
}
```

## References

- [MONAI Framework](https://monai.io/)
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) - Rombach et al.
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling

______________________________________________________________________

## Contact

For questions or issues:

- **Author**: Tuong Vy PHAM (tv.pham1996@gmail.com)
- **Issues**: Please open an issue on the repository
