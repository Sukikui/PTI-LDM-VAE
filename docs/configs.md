# Configs

This page focuses on what you actually change in practice, how configs connect to each other, and the main gotchas.
It avoids repeating every key already documented inside the JSON files.

## 1. Where configs live and how they are parsed
All configs are in `config/`.

> [!NOTE]
> In VAE configs, `_comment` fields are ignored and only serve as inline documentation.

## 2. Dependency graph (who depends on what)
The pipeline is sequential:

```
VAE config + checkpoint
        ↓
Regression head config + checkpoint
        ↓
LDM config
```

Concretely:
- Regression head config must point to a **VAE config + VAE checkpoint**.
- LDM config must point to a **VAE config + checkpoint** and a **regression head config + checkpoint**.

## 3. The few fields you typically edit
These are the fields you almost always change when starting a new run.

### VAE
- `data_base_dir` (e.g. `data/train_val`)
- `run_dir` (where outputs are stored)
- `data_source` (`edente`, `dente`, or `both`)
- `train_split` or `val_dir`
- `autoencoder_train.batch_size`, `autoencoder_train.lr`, `autoencoder_train.max_epochs`
- `autoencoder_train.recon_loss` (`l1` / `l2`)
- `autoencoder_train.adv_enabled` (adversarial on/off)
- `regularized_attributes.enabled` (AR-VAE on/off)

Example: [`config/vae_both_no_adv.json`](../config/vae_both_no_adv.json)

### Regression head
- `data.data_base_dir`, `data.data_source`
- `data.attributes_path` (metrics JSON)
- `targets` (ordered list of metrics)
- `vae.checkpoint` (must match your VAE run)
- `run_dir`
- `regression_train.batch_size`, `regression_train.lr`, `regression_train.max_epochs`
- `regression_train.target_norm` (`none` or `standard`)

Example:
- [`config/nreg_edente_from_both.json`](../config/nreg_edente_from_both.json)

### LDM
- `data.data_base_dir` (paired data root)
- `run_dir`
- `vae.checkpoint` and `regressor.checkpoint`
- `regressor.targets` (must match regression head)
- `conditioning.use_dentate_latent`, `conditioning.concat_dentate`
- `noise_init.*` (init mode + gradient noise params)
- `diffusion.num_train_timesteps`
- `train.batch_size`, `train.lr`, `train.max_epochs`

Example: [`config/ldm_both_no_adv_metrics_only_noisy.json`](../config/ldm_both_no_adv_metrics_only_noisy.json)

## 4. Interactions and gotchas (from code)
These details come directly from how the code reads configs:

### 4.1 AR-VAE: `attribute_latent_mapping` is required when enabled
When `regularized_attributes.enabled` is true, the code expects
`regularized_attributes.attribute_latent_mapping` to be present and non-empty.
If it is missing, VAE training fails.

### 4.2 Regression head: target order is strict
`targets` ordering is saved inside the checkpoint and validated at load time.
If the order changes, loading will fail.

> [!WARNING]
> Keep `targets` identical between training and inference (names + order).

### 4.3 LDM: UNet in/out channels are overwritten
Even if you set `unet.in_channels` and `unet.out_channels` in JSON, training will **override** them:
- `in_channels = latent_channels * (2 if concat_dentate else 1)`
- `out_channels = latent_channels`

So the real driver is the VAE latent size + `conditioning.concat_dentate`.

### 4.4 LDM: `use_dentate_latent` forces `concat_dentate`
If `conditioning.use_dentate_latent` is `false`, the code forces `concat_dentate` to `false`.

### 4.5 LDM: Choosing `noise_init.noise_weight`
When you use `noise_init`, the best way to pick `noise_weight` is to use the visualization tool:
```
python -m pti_ldm_vae_v2.analysis.visualize_noisy_latent \
  -c config/ldm_both_no_adv_metrics_only_noisy.json \
  --input-path data/train_val/dente/<file>.tif \
  --scale-factor <value>
```

Use the UI to inspect:
- the VAE latent channels,
- the noisy latent that feeds the LDM,
- and the VAE reconstruction from that noisy latent.

This makes it easy to pick a noise weight that is visible but not destructive.
The same `noise_init` block is used for training **and** sampling, so behavior stays aligned.
