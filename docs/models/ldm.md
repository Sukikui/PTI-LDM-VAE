# Latent Diffusion Model (LDM)

Minimal training + sampling pipeline for latent diffusion conditioned on dentate latents and/or metrics.

## Entry Points

```bash
python -m pti_ldm_vae_v2.ldm.train -c config/ldm_both_no_adv_metrics_only_noisy.json
python -m pti_ldm_vae_v2.ldm.sample \
  -c config/ldm_both_no_adv_metrics_only_noisy.json \
  --checkpoint runs/ldm_both_no_adv_metrics_only_noisy/trained_weights/ldm_unet_best.pth \
  --input-dir data/test/dente
```

## CLI Arguments

### Train

- `-c, --config-file` (required)
- `--max-epochs` (optional override)
- `--batch-size` (optional override)
- `--lr` (optional override)

### Sample

- `-c, --config-file` (required)
- `--checkpoint` (required)
- `--input-dir` (required)
- `--edente-dir` (optional override for ground-truth folder)
- `--output-dir` (optional override)
- `--num-steps` (default: 50)
- `--eta` (default: 0.0)
- `--guidance-scale` (optional)
- `--drop-z`, `--drop-metrics` (optional conditioning dropout)
- `--batch-size` (default: 4)
- `--num-samples` (optional cap)

## Related Scripts (useful for LDM)

Noisy latent visualization (to tune `noise_init` / `noise_weight`):

```bash
python -m pti_ldm_vae_v2.analysis.visualize_noisy_latent \
  -c config/ldm_both_no_adv_metrics_only_noisy.json \
  --input-path data/test/dente/example.tif \
  --scale-factor 160
```

Metric scatter plots after sampling:

```bash
python -m pti_ldm_vae_v2.analysis.plot_metric_ratios \
  --edente-metrics data/test/metrics/attributes_edente.json \
  --dente-metrics data/test/metrics/attributes_dente.json \
  --pred-metrics runs/ldm_both_no_adv_metrics_only_noisy/metrics/data/test/dente/attributes_edente_pred.json
```

Ground-truth metric generation:

```bash
python -m pti_ldm_vae_v2.tools.compute_mask_metrics \
  --edente-dir data/test/edente \
  --dente-dir data/test/dente \
  --output-edente data/test/metrics/attributes_edente.json \
  --output-dente data/test/metrics/attributes_dente.json
```

## Outputs

### Training

- `run_dir/` comes from the config.
- `run_dir/trained_weights/` contains:
  - `ldm_unet_best.pth`
  - `ldm_unet_last.pth`
- `run_dir/splits/ldm_pairs.json` stores the train/val file lists.

### Sampling

Default output root:

```
<run_dir>/results/<input_dir_relative>
```

Files:
- `results_tif/` (predicted edentulous image only, rotated 90° clockwise)
- `results_png/` (triplet view: dentate | edentulous_gt | edentulous_pred)

Metrics:

```
<run_dir>/metrics/<input_dir_relative>/attributes_edente_pred.json
```

Notes:
- `results_tif/` is rotated to match the dataset orientation.
- `results_png/` is not rotated and reflects raw model orientation.
- Metrics are computed from the rotated predictions using a binarization
  threshold of `0.2` and largest-component cleanup.

## Storage Layout (LDM)

Project layout (LDM only):

```
PTI-LDM-VAE/
├─ src/pti_ldm_vae_v2/ldm/             # LDM code (train/sample + helpers)
└─ config/
   └─ *.json                          # configs; each one sets run_dir
```

Run layout (created under `run_dir` from the config):

```
<run_dir>/
├─ trained_weights/
│  ├─ ldm_unet_last.pth
│  └─ ldm_unet_best.pth
├─ splits/
│  └─ ldm_pairs.json
├─ results/<input_dir_relative>/
│  ├─ results_tif/
│  └─ results_png/
├─ metrics/<input_dir_relative>/
│  └─ attributes_edente_pred.json
└─ wandb/                              # local W&B files (if enabled)
```

If `--output-dir` is passed for sampling, results go there instead.
