# Scripts

Training, inference, and analysis scripts for the PTI-LDM-VAE project.

## Quick Reference

| Script                    | Purpose                                                                         |
| ------------------------- | ------------------------------------------------------------------------------- |
| `train_vae.py`            | Train a Variational Autoencoder                                                 |
| `train_ldm.py`            | Train a conditional Latent Diffusion Model                                      |
| `inference_vae.py`        | Run VAE inference on images                                                     |
| `inference_ldm.py`        | Generate images with trained LDM                                                |
| `analyze_static.py`       | Generate static latent space visualizations                                     |
| `analyze_interactive.py`  | Interactive latent space exploration (UMAP/t-SNE, clic image, distance latente) |
| `compute_mask_metrics.py` | Measure edente/dente mask widths/heights                                        |

______________________________________________________________________

## Training Scripts

### train_vae.py

Train a VAE for image reconstruction.

**Basic usage:**

```bash
python scripts/train_vae.py -c config/vae_config.json
```

**With overrides:**

```bash
python scripts/train_vae.py \
  -c config/vae_config.json \
  --batch-size 16 \
  --lr 5e-5 \
  --max-epochs 50
```

**Multi-GPU training:**

```bash
torchrun --nproc_per_node=4 scripts/train_vae.py \
  -c config/vae_config.json \
  -g 4
```

**Arguments:**

- `-c, --config-file`: Path to unified config file (default: `./config/vae_config.json`)
- `-g, --gpus`: Number of GPUs (default: 1)
- `--batch-size`: Override batch size from config
- `--lr`: Override learning rate from config
- `--max-epochs`: Override max epochs from config
- `--num-workers`: Dataloader workers (default: 4)
- `--cache-rate`: RAM caching 0.0-1.0 (default: 0.0)
- `--seed`: Random seed (default: 42)
- `--subset-size`: Use only N images for debugging

**Outputs:**

- `<run_dir>/trained_weights/` - Model checkpoints
- `<run_dir>/tfevent/` - TensorBoard logs
- `<run_dir>/validation_samples/` - Validation images

______________________________________________________________________

### train_ldm.py

Train a conditional LDM for image-to-image translation.

**Basic usage:**

```bash
python scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1
```

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

- `<run_dir>/trained_weights/` - Model checkpoints
- `<run_dir>/tfevent/` - TensorBoard logs
- `<run_dir>/validation_samples/` - Generated images

______________________________________________________________________

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

- `-c, --config-file`: Path to config JSON (default: `./config/config_train_16g_cond.json`)
- `--checkpoint`: Path to VAE checkpoint (required)
- `--input-dir`: Directory with input TIF images (required)
- `--output-dir`: Output directory (default: auto-generated)
- `--num-samples`: Number of samples to process (default: all)
- `--batch-size`: Batch size (default: 8)

**Outputs:**

- `results_tif/` - Raw TIF files (original | reconstruction)
- `results_png/` - PNG files normalized for visualization

______________________________________________________________________

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

- `-e, --environment-file`: Path to environment JSON (default: `./config/environment_tif.json`)
- `-c, --config-file`: Path to config JSON (default: `./config/config_train_16g_cond.json`)
- `--checkpoint`: Path to LDM checkpoint (required)
- `--output-dir`: Output directory (default: auto-generated)
- `--num-samples`: Number of samples to generate (default: 10)
- `--batch-size`: Batch size (default: 1)

**Requirements:**

- Pretrained VAE checkpoint (specified in environment file)

**Outputs:**

- `results_tif/` - Raw TIF files (condition | target | synthetic)
- `results_png/` - PNG files normalized for visualization

______________________________________________________________________

### compute_mask_metrics.py

Compute geometric attributes (height and width) for paired edente/dente masks. Edente widths are sampled evenly top-to-bottom inside each bounding box, and dente widths are measured relative to fixed heights specified in millimeters.

**Usage:**

```bash
python scripts/compute_mask_metrics.py \
  --edente-dir data/edente \
  --dente-dir data/dente \
  --output-edente data/attributes_edente.json \
  --output-dente data/attributes_dente.json \
  --pixel-size-mm 0.15 \
  --dente-heights-mm 10 14 18 22 \
  --edente-width-samples 5
```

**Arguments:**

- `--edente-dir`: Directory containing edente masks (default: `./data/edente`).
- `--dente-dir`: Directory containing dente masks (default: `./data/dente`).
- `--output-edente`: Output JSON path for edente metrics (default: `./data/attributes_edente.json`).
- `--output-dente`: Output JSON path for dente metrics (default: `./data/attributes_dente.json`).
- `--pixel-size-mm`: Pixel size in millimeters used to convert dente heights (default: `0.15`).
- `--dente-heights-mm`: Heights in millimeters (from the bottom) at which dente widths are measured (default: `5 10 14 18 22`).
- `--edente-width-samples`: Number of evenly spaced widths across the edente bounding box (default: `5`).

**Outputs:**

- Two JSON files keyed by mask filename. Each entry contains the edente bounding-box height, the list of `width_i` measurements, and for dente masks the requested physical heights plus pixel widths at those locations.

______________________________________________________________________

## Analysis Scripts

### analyze_static.py

Generate static high-resolution latent space visualizations (UMAP or t-SNE).

**Basic usage (UMAP):**

```bash
python scripts/analyze_static.py \
  --vae-weights runs/vae_baseline/trained_weights/autoencoder_epoch73.pth \
  --config-file config/vae_config.json \
  --folder-edente data/edente/ \
  --folder-dente data/dente/ \
  --output-dir results/umap_analysis \
  --method umap \
  --max-images 1000
```

**t-SNE example:**

```bash
python scripts/analyze_static.py \
  --vae-weights runs/vae_baseline/trained_weights/autoencoder_epoch73.pth \
  --config-file config/vae_config.json \
  --folder-edente data/edente/ \
  --folder-dente data/dente/ \
  --output-dir results/tsne_analysis \
  --method tsne \
  --perplexity 30 \
  --max-images 1000
```

**Arguments:**

- `--vae-weights`: Path to trained VAE weights (required)
- `--config-file`: Path to config JSON (required)
- `--folder-edente`: Path to edentulous images (required)
- `--folder-dente`: Path to dental images (optional)
- `--output-dir`: Output directory (required)
- `--method`: Projection method: `umap` or `tsne` (default: `umap`)
- `--max-images`: Max images per group (default: 1000)
- `--color-by-patient`: Color points by patient ID
- `--n-neighbors`: UMAP n_neighbors (default: 40)
- `--min-dist`: UMAP min_dist (default: 0.5)
- `--perplexity`: t-SNE perplexity (default: 30)
- `--subtitle`: Custom subtitle for plot
- `--dpi`: Image DPI (default: 300)
- `--seed`: Random seed (default: 42)

**Outputs:**

- `{method}_projection.html` - Interactive Plotly visualization
- `{method}_projection.png` - High-resolution static image
- `color_legend.txt` - Color mapping (if `--color-by-patient`)
- `distance_metrics.txt` - Distance statistics (two-group mode)
- `exams_sorted_by_distance.txt` - Patients sorted by distance (two-group mode)

______________________________________________________________________

### analyze_interactive.py

Interactive latent space exploration with web server and image viewer.

**Usage:**

```bash
python scripts/analyze_interactive.py \
  --vae-weights runs/vae_baseline/trained_weights/autoencoder_epoch73.pth \
  --config-file config/vae_config.json \
  --folder-edente data/edente/ \
  --folder-dente data/dente/ \
  --method tsne \
  --max-images 500
```

**Arguments:**
Same as `analyze_static.py`, minus `--dpi` and `--output-dir`.

**Features:**

- Interactive scatter plot
- Click on points to view images
- Real-time exploration
- Web-based interface

______________________________________________________________________

## Data Directory Structure

Your data should be organized as follows:

```
data_base_dir/
├── edente/          # Edentulous images
│   ├── image_001.tif
│   ├── image_002.tif
│   └── ...
└── dente/           # Dental images (for LDM only)
    ├── image_001.tif
    ├── image_002.tif
    └── ...
```

**Notes:**

- For VAE training, only `edente/` folder is required
- For LDM training, both folders are required with matching filenames
- Images must be TIF format, float32, single channel

______________________________________________________________________

## Complete Workflow Example

### 1. Train VAE

```bash
python scripts/train_vae.py -c config/vae_config.json
```

### 2. Test VAE Inference

```bash
python scripts/inference_vae.py \
  --checkpoint runs/vae_baseline/trained_weights/autoencoder_epoch73.pth \
  --input-dir data/edente/ \
  --num-samples 10
```

### 3. Visualize Latent Space

```bash
python scripts/analyze_static.py \
  --vae-weights runs/vae_baseline/trained_weights/autoencoder_epoch73.pth \
  --config-file config/vae_config.json \
  --folder-edente data/edente/ \
  --output-dir results/latent_viz \
  --method umap
```

### 4. Train LDM

Update `environment_tif.json` with VAE checkpoint path, then:

```bash
python scripts/train_ldm.py \
  -e config/environment_tif.json \
  -c config/config_train_16g_cond.json \
  -g 1
```

### 5. Generate Images with LDM

```bash
python scripts/inference_ldm.py \
  --checkpoint runs/ldm_experiment/trained_weights/checkpoint_epoch50.pth \
  --num-samples 20
```

______________________________________________________________________

## Monitoring Training

All training scripts log to TensorBoard:

```bash
tensorboard --logdir runs/
```

Then open http://localhost:6006 in your browser.

**Available visualizations:**

- Loss curves (train/validation)
- Image triplets (original | reconstruction | difference)
- Generated samples during training

______________________________________________________________________

## Configuration

### VAE Training

Uses a single unified configuration file. See `config/README.md` for details.

**Example: `config/vae_config.json`**

```json
{
  "data_base_dir": "./data",
  "run_dir": "./runs/vae_baseline",
  "data_source": "edente",
  "train_split": 0.9,
  "autoencoder_def": { ... },
  "autoencoder_train": {
    "batch_size": 8,
    "lr": 2.5e-5,
    "max_epochs": 100
  }
}
```

### LDM Training

Uses two configuration files:

- `environment_tif.json` - Paths and VAE checkpoint
- `config_train_16g_cond.json` - Model architecture and hyperparameters

______________________________________________________________________

## Notes

- All scripts use deterministic seeding (seed=42) by default
- Training checkpoints include optimizer states for resuming
- Best models are saved based on validation loss
- Images are rotated 90° counterclockwise (k=3) for correct orientation
- Patient IDs are extracted from filenames: `<slice_id>_<date>_<patient_id>.tif`
