# Scripts

This directory contains training and inference scripts for the PTI-LDM-VAE project.

## Overview

- **`train_vae.py`**: Train a Variational Autoencoder (VAE)
- **`train_ldm.py`**: Train a conditional Latent Diffusion Model (LDM)
- **`inference_vae.py`**: Run inference with a trained VAE
- **`inference_ldm.py`**: Generate images with a trained LDM

---

## Training Scripts

### train_vae.py

Train a VAE for image reconstruction.

**Usage:**

```bash
python scripts/train_vae.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1
```

**Arguments:**
- `-e, --environment-file`: Path to environment JSON file (default: `./config/environment_tif.json`)
- `-c, --config-file`: Path to config JSON file (default: `./config/config_train_16g_cond.json`)
- `-g, --gpus`: Number of GPUs to use (default: 1)

**Multi-GPU training:**

```bash
torchrun --nproc_per_node=4 scripts/train_vae.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 4
```

**Outputs:**
- Checkpoints saved in `<run_dir>/trained_weights/`
- TensorBoard logs in `<run_dir>/tfevent/`
- Validation samples in `<run_dir>/validation_samples/`

---

### train_ldm.py

Train a conditional LDM for image-to-image translation.

**Usage:**

```bash
python scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1
```

**Arguments:**
- `-e, --environment-file`: Path to environment JSON file (default: `./config/environment_tif.json`)
- `-c, --config-file`: Path to config JSON file (default: `./config/config_train_16g_cond.json`)
- `-g, --gpus`: Number of GPUs to use (default: 1)

**Multi-GPU training:**

```bash
torchrun --nproc_per_node=4 scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 4
```

**Requirements:**
- Pretrained VAE checkpoint (specified in environment file)

**Outputs:**
- Checkpoints saved in `<run_dir>/trained_weights/`
- TensorBoard logs in `<run_dir>/tfevent/`
- Validation samples in `<run_dir>/validation_samples/`

---

## Inference Scripts

### inference_vae.py

Run inference on images using a trained VAE.

**Usage:**

```bash
python scripts/inference_vae.py \
  --checkpoint path/to/checkpoint_epoch73.pth \
  --input-dir path/to/images/ \
  --output-dir inference_results/ \
  --num-samples 20 \
  --batch-size 8
```

**Arguments:**
- `-c, --config-file`: Path to config JSON file (default: `./config/config_train_16g_cond.json`)
- `--checkpoint`: Path to VAE checkpoint file (required)
- `--input-dir`: Directory containing input TIF images (required)
- `--output-dir`: Output directory (default: `inference_vae_<checkpoint_name>`)
- `--num-samples`: Number of samples to process (default: all)
- `--batch-size`: Batch size (default: 8)

**Outputs:**
- `results_tif/`: Raw TIF files (original | reconstruction)
- `results_png/`: PNG files normalized for visualization

---

### inference_ldm.py

Generate images using a trained LDM with conditioning.

**Usage:**

```bash
python scripts/inference_ldm.py \
  --checkpoint path/to/checkpoint_epoch50.pth \
  --output-dir inference_ldm_results/ \
  --num-samples 10 \
  --batch-size 1
```

**Arguments:**
- `-e, --environment-file`: Path to environment JSON file (default: `./config/environment_tif.json`)
- `-c, --config-file`: Path to config JSON file (default: `./config/config_train_16g_cond.json`)
- `--checkpoint`: Path to LDM checkpoint file (required)
- `--output-dir`: Output directory (default: `inference_<checkpoint_name>`)
- `--num-samples`: Number of samples to generate (default: 10)
- `--batch-size`: Batch size (default: 1)

**Requirements:**
- Pretrained VAE checkpoint (specified in environment file)

**Outputs:**
- `results_tif/`: Raw TIF files (condition | target | synthetic)
- `results_png/`: PNG files normalized for visualization

---

## Configuration Files

### Environment File (environment_tif.json)

Contains paths and environment-specific settings:

```json
{
  "vae": {
    "data_base_dir": "./data/",
    "run_dir": "./runs/vae_experiment/",
    "resume_ckpt": false,
    "checkpoint_dir": ""
  },
  "ldm": {
    "data_base_dir": "./data/",
    "run_dir": "./runs/ldm_experiment/",
    "autoencoder_path": "./runs/vae_experiment/trained_weights/autoencoder_epoch73.pth",
    "resume_ckpt": false,
    "checkpoint_dir": ""
  }
}
```

### Config File (config_train_16g_cond.json)

Contains model architecture and training hyperparameters:

```json
{
  "spatial_dims": 2,
  "image_channels": 1,
  "latent_channels": 4,
  "augment": false,
  "autoencoder_def": {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "latent_channels": 4,
    "channels": [64, 128, 256],
    "num_res_blocks": 2,
    "norm_num_groups": 32,
    "attention_levels": [false, false, false],
    "with_encoder_nonlocal_attn": true,
    "with_decoder_nonlocal_attn": true
  },
  "autoencoder_train": {
    "batch_size": 8,
    "patch_size": [256, 256],
    "lr": 2.5e-5,
    "perceptual_weight": 1.0,
    "kl_weight": 1e-6,
    "recon_loss": "l1",
    "max_epochs": 100,
    "val_interval": 1
  },
  "diffusion_def": {
    "spatial_dims": 2,
    "in_channels": 4,
    "out_channels": 4,
    "channels": [32, 64, 128, 256],
    "attention_levels": [false, true, true, true],
    "num_head_channels": [0, 32, 32, 32],
    "num_res_blocks": 2,
    "with_conditioning": true,
    "cross_attention_dim": 512
  },
  "diffusion_train": {
    "batch_size": 8,
    "patch_size": [256, 256],
    "lr": 1e-5,
    "max_epochs": 51,
    "val_interval": 2,
    "lr_scheduler_milestones": [1000]
  },
  "NoiseScheduler": {
    "num_train_timesteps": 1000,
    "beta_start": 0.0015,
    "beta_end": 0.0195
  }
}
```

---

## Data Directory Structure

Your data should be organized as follows:

```
data_base_dir/
├── edente/          # Target images (edentulous)
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
└── dente/           # Condition images (dental) - for LDM only
    ├── image_001.tif
    ├── image_002.tif
    └── ...
```

**Notes:**
- For VAE training, only `edente/` folder is required
- For LDM training, both `edente/` and `dente/` folders are required
- Image names must match between folders for paired training

---

## Complete Workflow Example

### 1. Train VAE

```bash
python scripts/train_vae.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1
```

### 2. Test VAE Inference

```bash
python scripts/inference_vae.py \
  --checkpoint runs/vae_experiment/trained_weights/checkpoint_epoch73.pth \
  --input-dir data/edente/ \
  --num-samples 10
```

### 3. Train LDM

Update `environment_tif.json` with VAE checkpoint path, then:

```bash
python scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1
```

### 4. Generate Images with LDM

```bash
python scripts/inference_ldm.py \
  --checkpoint runs/ldm_experiment/trained_weights/checkpoint_epoch50.pth \
  --num-samples 20
```

---

## Monitoring Training

All scripts log to TensorBoard. To visualize:

```bash
tensorboard --logdir runs/
```

Then open http://localhost:6006 in your browser.

---

## Notes

- All scripts use deterministic seeding (seed=42) for reproducibility
- Training checkpoints include optimizer states for resuming
- Best models are saved based on validation loss
- Images are rotated 90° counterclockwise (k=3) for correct orientation

---

## Analysis Scripts

### analyze_umap.py

Analyze VAE latent space using UMAP dimensionality reduction.

**Usage (single group):**

```bash
python scripts/analyze_umap.py \
  --vae-weights runs/vae_experiment/trained_weights/autoencoder_epoch73.pth \
  --config-file config/config_train_16g_cond.json \
  --folder-group1 data/edente/ \
  --output-dir analysis/umap_edente/ \
  --max-images 1000 \
  --color-by-exam
```

**Usage (two-group comparison):**

```bash
python scripts/analyze_umap.py \
  --vae-weights runs/vae_experiment/trained_weights/autoencoder_epoch73.pth \
  --config-file config/config_train_16g_cond.json \
  --folder-group1 data/edente/ \
  --folder-group2 data/dente/ \
  --output-dir analysis/umap_comparison/ \
  --max-images 1000 \
  --color-by-exam \
  --n-neighbors 40 \
  --min-dist 0.5
```

**Arguments:**
- `--vae-weights`: Path to trained VAE weights (required)
- `--config-file`: Path to config JSON file (required)
- `--folder-group1`: Path to first image group (required)
- `--folder-group2`: Path to second image group (optional)
- `--output-dir`: Output directory for results (required)
- `--max-images`: Maximum images per group (default: 1000)
- `--patch-size`: Image patch size H W (default: 256 256)
- `--color-by-exam`: Color points by exam ID instead of group
- `--n-neighbors`: UMAP n_neighbors parameter (default: 40)
- `--min-dist`: UMAP min_dist parameter (default: 0.5)
- `--seed`: Random seed (default: 42)

**Outputs:**
- `umap_projection.png` - 2D visualization of latent space
- `color_legend.txt` - Color mapping for exams (if --color-by-exam)
- `distance_metrics.txt` - Distance statistics per exam (two-group mode)
- `exams_sorted_by_distance.txt` - Exams sorted by latent distance (two-group mode)

---

### analyze_tsne.py

Analyze VAE latent space using t-SNE dimensionality reduction.

**Usage (single group):**

```bash
python scripts/analyze_tsne.py \
  --vae-weights runs/vae_experiment/trained_weights/autoencoder_epoch73.pth \
  --config-file config/config_train_16g_cond.json \
  --folder-group1 data/edente/ \
  --output-dir analysis/tsne_edente/ \
  --max-images 1000 \
  --color-by-exam
```

**Usage (two-group comparison):**

```bash
python scripts/analyze_tsne.py \
  --vae-weights runs/vae_experiment/trained_weights/autoencoder_epoch73.pth \
  --config-file config/config_train_16g_cond.json \
  --folder-group1 data/edente/ \
  --folder-group2 data/dente/ \
  --output-dir analysis/tsne_comparison/ \
  --max-images 1000 \
  --perplexity 30
```

**Arguments:**
- `--vae-weights`: Path to trained VAE weights (required)
- `--config-file`: Path to config JSON file (required)
- `--folder-group1`: Path to first image group (required)
- `--folder-group2`: Path to second image group (optional)
- `--output-dir`: Output directory for results (required)
- `--max-images`: Maximum images per group (default: 1000)
- `--patch-size`: Image patch size H W (default: 256 256)
- `--color-by-exam`: Color points by exam ID instead of group
- `--perplexity`: t-SNE perplexity parameter (default: 30)
- `--seed`: Random seed (default: 42)

**Outputs:**
- `tsne_projection.png` - 2D visualization of latent space
- `color_legend.txt` - Color mapping for exams (if --color-by-exam)
- `distance_metrics.txt` - Distance statistics per exam (two-group mode)
- `exams_sorted_by_distance.txt` - Exams sorted by latent distance (two-group mode)

**Notes:**
- Image filenames should follow the pattern: `<slice_number>_<exam_id>.tif`
- Example: `0001_patient_ABC_session1.tif`
- The exam ID (everything after first underscore) is used for grouping and coloring
- Analysis scripts use PCA (50 components) before UMAP/t-SNE for efficiency
- t-SNE computation may take several minutes for large datasets