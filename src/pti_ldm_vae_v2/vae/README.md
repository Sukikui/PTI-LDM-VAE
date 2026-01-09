# VAE Module

VAE training, inference, and evaluation logic.
Shared helpers live in `pti_ldm_vae_v2/vae_regression_common` and are the only cross-module dependency
(along with third-party libs).

## Entry Points

The wrappers in `vae_scripts/` call these modules:

```bash
python vae_scripts/train_vae.py -c config/vae_both_no_adv.json
python vae_scripts/inference_vae.py -c config/vae_both_no_adv.json --checkpoint runs/vae_both_no_adv/trained_weights/autoencoder_last.pt --input-dir data/test/edente
python vae_scripts/evaluate_vae.py -c config/vae_both_no_adv.json --checkpoint runs/vae_both_no_adv/trained_weights/autoencoder_last.pt --input-dir data/test/edente
```

Interactive latent analysis (implemented in `pti_ldm_vae_v2/analysis`, runs a local Dash app):

```bash
python -m pti_ldm_vae_v2.analysis.analyze_interactive \
  -c config/vae_both_no_adv.json \
  --checkpoint runs/vae_both_no_adv/trained_weights/autoencoder_last.pt \
  --folder-edente data/test/edente \
  --folder-dente data/test/dente \
  --method tsne
```

Static latent analysis (saves PNG/HTML + metrics):

```bash
python -m pti_ldm_vae_v2.analysis.analyze_static \
  -c config/vae_both_no_adv.json \
  --checkpoint runs/vae_both_no_adv/trained_weights/autoencoder_last.pt \
  --folder-edente data/test/edente \
  --folder-dente data/test/dente \
  --method umap
```

## CLI Arguments

### Train

- `-c, --config-file` (required)
- `--batch-size` (optional override)
- `--lr` (optional override)
- `--max-epochs` (optional override)
- `--seed` (default: 42)

### Inference

- `-c, --config-file` (required)
- `--checkpoint` (required)
- `--input-dir` (required)
- `--output-dir` (optional override)
- `--num-samples` (optional cap)
- `--batch-size` (default: 8)
- `--seed` (default: 42)

### Evaluation

- `-c, --config-file` (required)
- `--checkpoint` (required)
- `--input-dir` (required)
- `--output-dir` (optional override)
- `--num-samples` (optional cap)
- `--batch-size` (default: 8)
- `--seed` (default: 42)

### Interactive Latent Analysis

- `-c, --config-file` (required)
- `--checkpoint` (required)
- `--folder-edente` (required)
- `--folder-dente` (optional)
- `--num-samples` (default: 3000)
- `--method` (default: tsne, choices: `umap`/`tsne`)
- `--debug` (optional)

### Static Latent Analysis

- `-c, --config-file` (required)
- `--checkpoint` (required)
- `--folder-edente` (required)
- `--folder-dente` (optional)
- `--output-dir` (optional override)
- `--num-samples` (default: 1000)
- `--method` (default: umap, choices: `umap`/`tsne`)
- `--n-neighbors` (default: 40, UMAP)
- `--min-dist` (default: 0.5, UMAP)
- `--perplexity` (default: 30, t-SNE)
- `--dpi` (default: 300)
- `--subtitle` (optional)

## Outputs

### Training

- `run_dir/` comes from the config.
- `run_dir/trained_weights/` contains:
  - `autoencoder_last.pt`, `discriminator_last.pt` (latest)
  - `autoencoder_epoch*.pth`, `discriminator_epoch*.pth`, `checkpoint_epoch*.pth` (best)
- `run_dir/splits/vae_split.json` stores the train/val file lists.
- Validation images are logged only to W&B (no local image dumps).

### Inference

Default output root:

```
<run_dir>/inference/<input_dir_relative>
```

Files:
- `results_tif/` (original | reconstruction, concatenated)
- `results_png/` (display-normalized)

If `--output-dir` is provided, the outputs go there instead.

### Evaluation

Default output:

```
<run_dir>/eval/<input_dir_relative>/metrics.json
```

`metrics.json` contains the aggregated metrics and the list of evaluated files.

## Storage Layout (VAE)

Project layout (VAE only):

```
pti-ldm-vae/
├─ src/pti_ldm_vae_v2/vae/           # VAE code (train/infer/eval + helpers)
├─ vae_scripts/
│  ├─ train_vae.py
│  ├─ inference_vae.py
│  └─ evaluate_vae.py
└─ config/
   └─ *.json                       # configs; each one sets run_dir
```

Shared helpers: `src/pti_ldm_vae_v2/vae_regression_common/`.
VAEModel lives in `src/pti_ldm_vae_v2/vae_regression_common/vae_model.py`.
Preprocess transform lives in `src/pti_ldm_vae_v2/vae_regression_common/transforms.py` (`build_preprocess_transform`).

Run layout (created under `run_dir` from the config):

```
<run_dir>/
├─ trained_weights/
│  ├─ autoencoder_last.pt
│  ├─ discriminator_last.pt
│  ├─ autoencoder_epoch*.pth
│  ├─ discriminator_epoch*.pth
│  └─ checkpoint_epoch*.pth
├─ splits/
│  └─ vae_split.json
└─ wandb/                          # local W&B files (if enabled)
```

Latent analysis cache (auto-created, speeds up re-runs):

```
<run_dir>/analysis/latents_cache/
```

Static analysis outputs (default):

```
<run_dir>/analysis/static/<folder_edente_relative>/
├─ umap_projection.png | tsne_projection.png
├─ color_legend.txt
├─ distance_metrics.txt
└─ exams_sorted_by_distance.txt
```

Inference outputs (default):

```
<run_dir>/inference/<input_dir_relative>/
├─ results_tif/
└─ results_png/
```

Evaluation outputs (default):

```
<run_dir>/eval/<input_dir_relative>/
└─ metrics.json
```

If `--output-dir` is passed for inference or evaluation, results go there instead.

## Key Config Fields

Minimum config keys used by this module:

- `run_dir`
- `data_base_dir`, `data_source`, `train_split`, `val_dir`
- `autoencoder_def` (architecture)
- `autoencoder_train`:
  - `batch_size`, `lr`, `max_epochs`, `patch_size`
  - `subset_size` (optional, limits the number of training samples)
  - `recon_loss`, `kl_weight`, `perceptual_weight`
  - `adv_enabled`, `adv_weight` (GAN branch)
  - `ar_vae_enabled`, `ar_vae_weight` (AR-VAE)
  - `val_interval`
- `spatial_dims`
- `regularized_attributes` (only if AR-VAE is enabled)
- `wandb` (if logging is enabled)

Notes:
- `regularized_attributes.attribute_file` can be a single JSON path or a mapping `{ "edente": "...", "dente": "..." }`.
  If `data_source` is `both` and a single path is provided, it is reused for both sources.

## Defaults

- `DEFAULT_NUM_WORKERS = 4` in `pti_ldm_vae_v2/vae_regression_common/runtime.py` (internal only, not a CLI arg).
