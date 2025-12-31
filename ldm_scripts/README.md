# LDM Scripts

Scripts pour entraîner et inférer le Latent Diffusion Model conditionné par l'image dentée et les métriques prédites par la tête de régression.

## Fichiers

- `train_ldm.py` : boucle d'entraînement (encode VAE gelé, MLP gelé, UNet conditionné, checkpoints).
- `sample_ldm.py` : génération à partir d'images dentées (encode VAE, prédit métriques, diffusion inverse, decode).
- `config/ldm_example.json` : exemple de configuration.

## Configuration

```json
{
  "run_dir": "./runs/ldm_example",
  "seed": 42,
  "data": {
    "data_base_dir": "./data/train_val",
    "patch_size": [256, 256],
    "train_split": 0.9,
    "num_workers": 4,
    "cache_rate": 0.0
  },
  "vae": {
    "config_file": "./config/vae_edente.json",
    "checkpoint": "./runs/vae_edente/trained_weights/autoencoder_epoch73.pth"
  },
  "regressor": {
    "config_file": "./config/reg_edente_from_dente.json",
    "checkpoint": "./runs/reg_head_edente/trained_weights/head_best.pth",
    "targets": ["height", "width_upper", "width_middle", "width_lower"]
  },
  "unet": {
    "spatial_dims": 2,
    "in_channels": 4,
    "out_channels": 4,
    "channels": [32, 64, 128],
    "attention_levels": [false, true, true],
    "num_head_channels": [0, 32, 32],
    "num_res_blocks": 2,
    "with_conditioning": true,
    "cross_attention_dim": 256
  },
  "conditioning": {
    "concat_dentate": false,
    "condition_dropout": 0.1,
    "metrics_dropout": 0.1,
    "metric_dropout": 0.0
  },
  "diffusion": {
    "num_train_timesteps": 1000,
    "beta_start": 0.00085,
    "beta_end": 0.012
  },
  "train": {
    "batch_size": 4,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "max_epochs": 100,
    "val_interval": 1,
    "log_interval": 1,
    "clip_grad": 1.0,
    "ema_decay": 0.999
  }
}
```

- `concat_dentate=false` utilise le denté dans le cross-attention uniquement (tokens latents + token métrique). Mettre à `true` si l'on veut concaténer `[z_t; z0*]` en entrée du UNet (l'in_channels sera ajusté automatiquement).
- `condition_dropout` / `metrics_dropout` appliquent un dropout sur les conditions pour activer le classifier-free guidance (guidance au sampling via `--guidance-scale`).
- `num_train_timesteps` définit la taille du schedule DDPM (betas linéaires `beta_start`→`beta_end`).

## Entraînement

```bash
python ldm_scripts/train_ldm.py -c config/ldm_example.json \
  --batch-size 4 \
  --lr 1e-4 \
  --max-epochs 100
```

Exemple direct avec tes poids : `config/ldm_both_no_adv.json` (VAE `checkpoint_epoch97.pth`, head `head_last.pth`).

- Checkpoints : `runs/<run_dir>/trained_weights/ldm_unet_last.pth` et `ldm_unet_best.pth` (+ EMA si activé).
  Ces checkpoints incluent aussi `metric_embed` et `condition_builder` pour garder le conditionnement coherent.
- Split sauvegardé : `runs/<run_dir>/splits/ldm_pairs.json`.

## Generation

```bash
python ldm_scripts/sample_ldm.py \
  -c config/ldm_example.json \
  --checkpoint runs/ldm_example/trained_weights/ldm_unet_best.pth \
  --input-dir data/dente \
  --num-steps 50 \
  --eta 0.0 \
  --guidance-scale 3.0
```

Sorties (par défaut) : `inference_<checkpoint_name>/results_tif/` (concat denté|édenté synth) et `results_png/` (version normalisée).

## Weights & Biases

- Active via le bloc `wandb` de la config (`enabled: true`). Le script logge :
  - `train/loss_total` et `train/noise_loss` par step.
  - `train/loss_total_epoch` et `train/noise_loss_epoch` par epoch.
  - `val/loss_total` et `val/noise_loss` par epoch.
  - `time/epoch` en secondes.
  - `best/val_loss_total` dans le résumé.
- Les clés `project`, `entity`, `name`, `tags`, `notes` suivent la même convention que les scripts VAE/régression. Si `wandb` est absent ou `enabled=false`, rien n’est loggé.
- Aucun TensorBoard n’est utilisé.
- Le script charge automatiquement le fichier `.env` (ex: `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`).

## Notes rapides

- VAE et tête de régression sont gelés (`encode_stage_2_inputs` pour la cible édentée, `encode_deterministic` pour le conditionnement denté).
- Metrics utilisées telles que prédites par la tête (espace normalisé si la tête a été entraînée ainsi).
- Scheduler simplifié DDPM linéaire ; sampling DDIM déterministe (`eta=0`), bruit stochastique si `eta>0`.
- Guidance : activer `condition_dropout`/`metrics_dropout` a l'entrainement puis utiliser `--guidance-scale` au sampling.
- Les scripts de generation chargent aussi les poids du conditionnement; si un ancien checkpoint ne les contient pas,
  un warning est affiche et les modules sont initialises par defaut.
