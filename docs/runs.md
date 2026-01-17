# Runs & Outputs

Every config defines a `run_dir` where artifacts are written. This page describes the exact folder layout per module.

## 1. How `run_dir` is chosen
All modules read `run_dir` from their JSON config.

## 2. Common convention `<input_dir_relative>`
Inference/eval outputs mirror the input directory under the run:

```
<run_dir>/<subdir>/<input_dir_relative>/
```

Example:
- `--input-dir data/test/dente`
- outputs go to: `<run_dir>/inference/data/test/dente/` (or `results/...` for LDM)

## 3. VAE outputs
```
runs/<vae_run>/
|-- trained_weights/
|   |-- autoencoder_last.pt
|   |-- discriminator_last.pt           (only if adversarial is enabled)
|   |-- autoencoder_epoch<best>.pth
|   |-- discriminator_epoch<best>.pth   (only if adversarial is enabled)
|   `-- checkpoint_epoch<best>.pth
|-- splits/
|   `-- vae_split.json
|-- inference/<input_dir_relative>/
|   |-- results_tif/
|   `-- results_png/
|-- eval/<input_dir_relative>/
|   `-- metrics.json
`-- wandb/ (if enabled)
```

> [!NOTE]
> - Inference `results_tif/` stores `input | reconstruction` concatenated.  
> - Inference `results_png/` is display-normalized.  
> - `metrics.json` contains aggregate metrics and evaluated filenames.

## 4. Regression head outputs
```
runs/<reg_run>/
|-- trained_weights/
|   |-- head_last.pth
|   |-- head_best.pth
|   `-- target_norm_stats.json          (only when `regression_train.target_norm = "standard"`)
|-- inference/<input_dir_relative>/
|   `-- predictions.json
|-- eval/<input_dir_relative>/
|   `-- metrics.json
`-- wandb/ (if enabled)
```

> [!NOTE]
> - `predictions.json` maps filenames to predicted metrics.  
> - `metrics.json` contains evaluation loss + per-target metrics.

## 5. LDM outputs
```
runs/<ldm_run>/
|-- trained_weights/
|   |-- ldm_unet_last.pth
|   `-- ldm_unet_best.pth
|-- splits/
|   `-- ldm_pairs.json
|-- results/<input_dir_relative>/
|   |-- results_tif/
|   `-- results_png/
|-- metrics/<input_dir_relative>/
|   `-- attributes_edente_pred.json
`-- wandb/ (if enabled)
```

> [!NOTE]
> - `results_tif/` stores the predicted edentulous image only, rotated 90° clockwise.  
> - `results_png/` shows `dentate | edentulous_gt | edentulous_pred` concatenated (display-normalized).
> - `attributes_edente_pred.json` mirrors the `compute_mask_metrics` schema for predicted edentulous masks.

## 6. Output overrides
Inference and evaluation scripts accept `--output-dir` to override the default run subfolder.

If you need config details, see [`docs/configs.md`](configs.md).
