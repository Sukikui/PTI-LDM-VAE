# W&B

This page explains how to enable logging, how settings are resolved, and which metrics are logged by each module.

## 1. Enable or disable logging
Each config has a `wandb` block:
```json
"wandb": {
  "enabled": true
}
```

Set `"enabled": false` to disable logging for that run.

## 2. Credentials and project settings
The project uses a `.env` file at the repo root. Start from the template:
```
cp .env.example .env
```

Then fill in the values (see comments inside `.env`):
```
WANDB_API_KEY=...
WANDB_ENTITY=your_username_or_team
WANDB_PROJECT=pti-project
```

Config fields (optional):
```json
"wandb": {
  "project": "pti-project",
  "entity": "your-entity",
  "name": "run_name",
  "tags": ["tag1", "tag2"],
  "notes": "short note"
}
```

Resolution order:
- `WANDB_PROJECT` overrides `wandb.project`.
- `WANDB_ENTITY` overrides `wandb.entity`.
- `wandb.name` overrides the default run name (`run_dir` folder name).

> [!TIP]
> Before starting a new experiment, update `wandb.name` in your config if you want a clean run name.

> [!NOTE]
> If the `wandb` package is not installed, scripts print a warning and continue without logging.

## 3. Offline mode
You can run offline by setting:
```
WANDB_MODE=offline
```

Tip: put it in `.env` for consistent behavior.

## 4. Local files
Local W&B files are written under:
```
<run_dir>/wandb/
```

## 5. Metrics by module
The lists below reflect the metrics actually logged by the code.

### 5.1 VAE
- **Train**: `train/recon_loss`, `train/kl_loss`, `train/perceptual_loss`, `train/adv_gen_loss`,
  `train/adv_disc_loss`, `train/loss_total`, `train/ar_loss_*` (if AR-VAE enabled).
- **Val**: `val/recon_loss`, `val/kl_loss`, `val/perceptual_loss`, `val/adv_gen_loss`,
  `val/adv_disc_loss`, `val/loss_total`, `val/ar_loss_*` (if AR-VAE enabled).
- **Images**: `train/triplets`, `val/triplets` (original | reconstruction | diff).
- **Timing**: `time_per_epoch`.

### 5.2 Regression head
- **Train**: `train/loss_mse` or `train/loss_huber` (depending on `regression_train.loss`).
- **Val**: `val/loss_mse` or `val/loss_huber`.
- **Metrics**: `val/mae`, `val/mse`, `val/r2` and per-target variants (`val/mae_<target>`, etc.).
- **Best**: `best/val_loss_*` (best validation loss so far).

> [!NOTE]
> The regression head always logs **MAE/MSE/R2**, even if the training loss is `smooth_l1`.
> Only one loss (`mse` or `smooth_l1`) is actually optimized, as defined by `regression_train.loss`.

### 5.3 LDM
- **Train (step)**: `train/loss_total`, `train/noise_loss`, `train/step`.
- **Train (epoch)**: `train/loss_total_epoch`, `train/noise_loss_epoch`.
- **Val**: `val/loss_total`, `val/noise_loss`.
- **Timing**: `time/epoch`.
