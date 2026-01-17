# VAE Module

VAE training, inference, and evaluation logic.
Shared helpers live in `pti_ldm_vae_v2/common`, and shared model definitions live in
`pti_ldm_vae_v2/models` (along with third-party libs).

## Entry Points

Train, inference, and evaluation entry points:

```bash
python -m pti_ldm_vae_v2.vae.train -c config/vae_both_no_adv.json
python -m pti_ldm_vae_v2.vae.infer \
  -c config/vae_both_no_adv.json \
  --checkpoint runs/vae_both_no_adv/trained_weights/autoencoder_last.pt \
  --input-dir data/test/edente
python -m pti_ldm_vae_v2.vae.eval \
  -c config/vae_both_no_adv.json \
  --checkpoint runs/vae_both_no_adv/trained_weights/autoencoder_last.pt \
  --input-dir data/test/edente
```

Checkpoints can be either:
- `autoencoder_last.pt` (latest)
- `autoencoder_epoch*.pth` (best epoch)
- `checkpoint_epoch*.pth` (full checkpoint with optimizer state)

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

AR channel viewer (single image, Dash):

```bash
python -m pti_ldm_vae_v2.analysis.analyze_ar_channels \
  -c config/vae_both_no_adv.json \
  --checkpoint runs/vae_both_no_adv/trained_weights/autoencoder_last.pt \
  --image-path data/test/edente/example.tif
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

### AR Channel Viewer

- `-c, --config-file` (required)
- `--checkpoint` (required)
- `--image-path` (required)
- `--host` (default: 127.0.0.1)
- `--port` (default: 8052)
- `--debug` (optional)

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

Notes:
- Filenames are sequential (`image0000.tif`, `image0000.png`) rather than input filenames.
- Ordering follows the sorted input list from `data/<source>/`.

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
PTI-LDM-VAE/
├─ src/pti_ldm_vae_v2/vae/           # VAE code (train/infer/eval + helpers)
└─ config/
   └─ *.json                       # configs; each one sets run_dir
```

Shared helpers: `src/pti_ldm_vae_v2/common/`.
VAEModel lives in `src/pti_ldm_vae_v2/models/vae.py`.
Preprocess transform lives in `src/pti_ldm_vae_v2/common/transforms.py` (`build_preprocess_transform`).

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
