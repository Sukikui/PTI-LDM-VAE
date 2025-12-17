# Tête de régression sur latents VAE

Ce modèle ajoute une tête MLP qui prédit les métriques d’attributs (celles utilisées pour l’AR-VAE) à partir des latents d’un VAE gelé. Les latents sont encodés « on the fly », aplatis intégralement, puis passés dans un MLP configurable.

______________________________________________________________________

## Structure et responsabilités

- `src/pti_ldm_vae/models/regression_head.py`
  - `LatentRegressor` : MLP configurable.
  - `VAELatentRegressor` : encode avec le VAE gelé (deterministic), aplatit le latent et applique le MLP.
- `src/pti_ldm_vae/data/dataloaders.py`
  - `create_regression_dataloaders(...)` : reprend le pipeline VAE (load/resize/norm/cache, sans augmentation) et associe les cibles depuis un JSON d’attributs pour retourner `(image, target_vector)`.
- `src/pti_ldm_vae/utils/regression_utils.py`
  - Normalisation standard des cibles, construction de la loss, boucles train/val, checkpoints dédiés et création du modèle (VAE gelé + tête) depuis la config (fichier centré régression mais factorisé pour garder les scripts concis).
- `src/pti_ldm_vae/utils/cli_common.py`
  - Helpers partagés : initialisation device/seed, résolution des dossiers d’output, chargement config + modèle VAE.
- `src/pti_ldm_vae/utils/metrics.py`
  - MAE/MSE par cible et agrégés.
- Scripts : `reg_scripts/train_regression.py`, `reg_scripts/evaluate_regression.py`, `reg_scripts/inference_regression.py`.
- Config exemple : `config/reg_edente_from_dente.json`.
- Docs : section “Regression head” dédiée dans ce README, clés W&B ajoutées dans `WANDB_METRICS.md`.

______________________________________________________________________

## Données et flux

1. Images chargées via le même preprocessing que le VAE (patch_size, normalisation locale, pas d’augmentation).
2. Cibles lues dans `attributes_path` (JSON mapping filename → dict de métriques), filtrées sur `targets`.
3. Batch : `(images, target_vectors)` → VAE encode → latent aplati → MLP → prédictions.
4. Perte régression (MSE ou Huber) sur les cibles; option de normalisation standard des cibles (stockage des stats pour inference/éval).
5. Logging : pertes + MAE/MSE par cible, W&B optionnel.

______________________________________________________________________

## Schéma de configuration (exemple `config/reg_edente_from_dente.json`)

```json
{
  "run_dir": "./runs/reg_head_edente",

  "data": {
    "data_base_dir": "./data/train_val/",
    "attributes_path": "./data/train_val/metrics/attributes_edente.json",
    "data_source": "edente",
    "train_split": 0.9,
    "val_dir": null,
    "patch_size": [256, 256],
    "cache_rate": 0.0,
    "num_workers": 4,
    "seed": 42,
    "subset_size": null,
    "normalize_attributes": null
  },

  "vae": {
    "config_file": "./config/ar_vae_edente.json",
    "checkpoint": "./runs/vae_baseline/trained_weights/autoencoder_epoch73.pth",
    "latent_agg": "flatten"
  },

  "targets": ["width_top", "width_mid", "width_bottom"],

  "regressor_def": {
    "hidden_dims": [256, 128],
    "dropout": 0.1,
    "activation": "relu"
  },

  "regression_train": {
    "batch_size": 16,
    "lr": 1e-4,
    "max_epochs": 50,
    "val_interval": 1,
    "target_norm": "standard",
    "loss": "mse"
  },

  "wandb": {
    "enabled": true,
    "name": "reg_head_edente",
    "tags": ["regression", "vae-latent"]
  }
}
```

Champs clés :

- `run_dir` : répertoire racine pour checkpoints/éval/inférence.
- `data.*` : chemins, splits, patch size et workers alignés avec les configs VAE.
- `targets` : liste des métriques à prédire et ordre de sortie du MLP.
- `vae` : config+checkpoint du VAE gelé, `latent_agg` reste `flatten`.
- `regression_train.target_norm` : `standard` pour centrer/réduire les cibles (stats sauvegardées), `none` pour brut.
- Overrides CLI : mêmes noms que pour le VAE (`--batch-size`, `--lr`, `--max-epochs`, `--num-workers`, `--cache-rate`, `--seed`, `--subset-size`), appliqués respectivement à `regression_train` et `data`.

______________________________________________________________________

## Scripts et usage

### Entraîner (`reg_scripts/train_regression.py`)

- Args principaux : `--config-file`, overrides `--batch-size`, `--lr`, `--max-epochs`, `--num-workers`, `--cache-rate`, `--seed`.
- Charge config, dataloaders via `create_regression_dataloaders`, VAE gelé via `load_vae_model`, tête `VAELatentRegressor`.
- Affiche un summary du modèle (latente aplatie, hidden_dims, #params) puis une barre de progression par époque.
- Si `wandb.enabled=true`, log `train/loss_<type>`, `val/loss_<type>` (mse ou huber selon la config), `val/mae*`, `val/mse*`, `best/val_loss_<type>` par époque.
- Sauvegardes : meilleurs/derniers poids de la tête (+ stats de normalisation si activées), éventuellement W&B run.
- Exemple :

```bash
python reg_scripts/train_regression.py \
  -c config/reg_edente_from_dente.json \
  --batch-size 8 \
  --lr 5e-5
```

### Évaluer (`reg_scripts/evaluate_regression.py`)

- Args : `--config-file`, `--checkpoint` (tête), `--input-dir`, `--batch-size`, `--num-workers`, `--num-samples`, `--output-dir`, `--seed`.
- Recharge VAE gelé + tête, dataloader identique; calcule MAE/MSE par cible et agrégés; écrit `metrics.json` dans `output-dir` (par défaut `<run_dir>/eval/`).
- Exemple :

```bash
python reg_scripts/evaluate_regression.py \
  -c config/reg_edente_from_dente.json \
  --checkpoint runs/reg_head/trained_weights/head_epoch20.pth \
  --input-dir data/edente/ \
  --output-dir evals/reg_head
```

### Inférer (`reg_scripts/inference_regression.py`)

- Args : `--config-file`, `--checkpoint` (tête), `--input-dir`, `--batch-size`, `--num-workers`, `--num-samples`, `--output-dir`, `--seed`.
- Même preprocessing; produit `predictions.json` (et/ou CSV) avec `{filename: {target: valeur}}` dans `output-dir` (par défaut `<run_dir>/inference/`).
- Exemple :

```bash
python reg_scripts/inference_regression.py \
  -c config/reg_edente_from_dente.json \
  --checkpoint runs/reg_head/trained_weights/head_epoch20.pth \
  --input-dir data/edente/ \
  --output-dir results/regression_preds
```

______________________________________________________________________

## Sorties attendues

Par défaut (si `run_dir` est défini dans la config), les artefacts sont regroupés sous ce répertoire.

```
runs/reg_head_edente/
├── trained_weights/
│   ├── head_last.pth
│   ├── head_epoch20.pth          # meilleur modèle sauvegardé
│   └── target_norm_stats.json    # si target_norm=standard
├── eval/
│   └── metrics.json              # MAE/MSE par cible + global + args + fichiers évalués
└── inference/
    └── predictions.json          # {filename: {target: valeur}}
```

______________________________________________________________________

## Points d’implémentation / robustesse

- **Validation des cibles** : lors de la création des dataloaders, vérifier que chaque fichier d’image a bien toutes les clés listées dans `targets` ; lever une erreur explicite sinon. Charger le JSON d’attributs une seule fois et partager en mémoire pour les workers.
- **Normalisation des cibles** : si `target_norm=standard`, sauvegarder `target_norm_stats.json` (moyenne/écart-type) avec l’ordre exact des `targets` pour détecter les incohérences entre train/éval/inférence.
- **run_dir par défaut** : si non fourni, utiliser `runs/<config_stem>` afin de conserver une arbo cohérente.
- **Dimension latente aplatie** : le flatten complet peut être volumineux ; un warning est émis si la dimension explose, incitant à réduire la taille du patch ou du VAE si nécessaire (pas de projection ajoutée).

______________________________________________________________________

## Points DRY

- Pipeline de données VAE réutilisé (pas de nouveau transform).
- Un seul loader VAE (`load_vae_model`) pour l’encodeur gelé.
- Helpers CLI partagés pour device/seed et sorties.
- Tête MLP isolée (`models/regression_head.py`) pour réutilisation.
