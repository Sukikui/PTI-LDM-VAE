# Regression Head Module

Training, inference, and evaluation logic for the regression head.
Shared helpers live in `pti_ldm_vae_v2/common`, and shared model definitions live in
`pti_ldm_vae_v2/models` (along with third-party libs).

## Entry Points

The wrappers in `reg_scripts/` call these modules:

```bash
python reg_scripts/train_regression.py -c config/reg_edente_from_dente.json
python reg_scripts/inference_regression.py -c config/reg_edente_from_dente.json --checkpoint runs/reg_head_edente/trained_weights/head_last.pth --input-dir data/test/edente
python reg_scripts/evaluate_regression.py -c config/reg_edente_from_dente.json --checkpoint runs/reg_head_edente/trained_weights/head_last.pth --input-dir data/test/edente
```

## CLI Arguments

### Train

- `-c, --config-file` (required)
- `--batch-size` (optional override)
- `--lr` (optional override)
- `--max-epochs` (optional override)
- `--seed` (optional override)
- `--resume-checkpoint` (optional checkpoint)

### Inference

- `-c, --config-file` (required)
- `--checkpoint` (required)
- `--input-dir` (required)
- `--output-dir` (optional override)
- `--num-samples` (optional cap)
- `--batch-size` (default: from config)
- `--seed` (default: 42)

### Evaluation

- `-c, --config-file` (required)
- `--checkpoint` (required)
- `--input-dir` (optional override; defaults to evaluation.data_base_dir)
- `--attributes-path` (optional override)
- `--output-dir` (optional override)
- `--num-samples` (optional cap)
- `--batch-size` (default: from config)
- `--seed` (default: 42)

## Outputs

### Training

- `run_dir/` comes from the config.
- `run_dir/trained_weights/` contains:
  - `head_last.pth` (latest)
  - `head_best.pth` (best validation)
  - `target_norm_stats.json` (if target_norm is standard)

### Inference

Default output root:

```
<run_dir>/inference/<input_dir_relative>
```

Files:
- `predictions.json` (mapping filename -> {target: value})

### Evaluation

Default output:

```
<run_dir>/eval/<input_dir_relative>/metrics.json
```

`metrics.json` contains the aggregated metrics and the list of evaluated files.

## Storage Layout (Regression Head)

Project layout (regression head only):

```
pti-ldm-vae/
|-- src/pti_ldm_vae_v2/regression_head/   # regression head code
|-- reg_scripts/
|   |-- train_regression.py
|   |-- inference_regression.py
|   `-- evaluate_regression.py
`-- config/
    `-- *.json                          # configs; each one sets run_dir
```

Shared helpers: `src/pti_ldm_vae_v2/common/`.
Models live in:
- `src/pti_ldm_vae_v2/models/vae.py` (VAEModel)
- `src/pti_ldm_vae_v2/models/regression_head.py` (LatentRegressor, VAELatentRegressor)
Preprocess transform lives in `src/pti_ldm_vae_v2/common/transforms.py` (`build_preprocess_transform`).

Run layout (created under `run_dir` from the config):

```
<run_dir>/
|-- trained_weights/
|   |-- head_last.pth
|   |-- head_best.pth
|   `-- target_norm_stats.json
|-- eval/
|   `-- <input_dir_relative>/
|       `-- metrics.json
`-- inference/
    `-- <input_dir_relative>/
        `-- predictions.json
```

If `--output-dir` is passed for inference or evaluation, results go there instead.

## Key Config Fields

Minimum config keys used by this module:

- `run_dir`
- `data`: `data_base_dir`, `attributes_path`, `data_source`, `train_split`, `val_dir`, `patch_size`, `cache_rate`,
  `num_workers`, `seed`, `subset_size`, `normalize_attributes`
- `evaluation`: `data_base_dir`, `attributes_path`, `data_source`, `patch_size`, `num_workers`, `normalize_attributes`
- `vae`: `config_file`, `checkpoint`
- `targets`
- `regressor_def`: `hidden_dims`, `dropout`, `activation`
- `regression_train`: `batch_size`, `lr`, `max_epochs`, `val_interval`, `target_norm`, `loss`, `weight_decay`
- `wandb` (if logging is enabled)
