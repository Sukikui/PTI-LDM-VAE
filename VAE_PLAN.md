# Plan d'entraînement VAE

Guide pour entraîner un VAE sur les datasets edente, dente, ou les deux combinés.

______________________________________________________________________

## Vue d'ensemble

**Trois configurations possibles** :

- **`edente`** : Images édentées uniquement (~6000 images)
- **`dente`** : Images dentées uniquement (~6000 images)
- **`both`** : Combinaison edente + dente (~12000 images)

______________________________________________________________________

## Étape 1 : Test rapide (5-10 minutes)

**Objectif** : Vérifier que tout fonctionne avant un long entraînement.

### Utiliser le config existant avec overrides

```bash
# Test edente (100 images, 5 epochs)
python scripts/train_vae.py \
  -c config/vae_config.json \
  --subset-size 100 \
  --max-epochs 5

# Pour tester dente ou both, créer vae_config_dente.json et vae_config_both.json
# en changeant simplement "data_source": "dente" ou "both"
```

### Vérifications

✅ **Données chargées correctement** (voir logs "Dataset Statistics")
✅ **Pas d'erreurs** (OOM, NaN loss, CUDA)
✅ **Loss diminue**
✅ **Images de validation créées** dans `runs/.../validation_samples/`
✅ **Checkpoints créés** dans `runs/.../trained_weights/`

______________________________________________________________________

## Étape 2 : Entraînement complet

### Créer les configs finaux

Copier `config/vae_config.json` vers trois nouveaux fichiers :

#### `config/vae_edente.json`

```json
{
  "data_base_dir": "./data",
  "run_dir": "./runs/vae_edente",
  "data_source": "edente",
  "train_split": 0.9,

  "autoencoder_train": {
    "batch_size": 8,
    "lr": 2.5e-5,
    "max_epochs": 100,
    // ... autres params identiques au config de base
  },

  "wandb": {
    "enabled": true,
    "name": "vae_edente",
    "tags": ["vae", "edente"]
  }
}
```

#### `config/vae_dente.json`

Identique mais avec `"data_source": "dente"` et `"run_dir": "./runs/vae_dente"`

#### `config/vae_both.json`

Identique mais avec `"data_source": "both"` et `"run_dir": "./runs/vae_both"`

### Lancer les entraînements

```bash
# Single GPU
python scripts/train_vae.py -c config/vae_edente.json
python scripts/train_vae.py -c config/vae_dente.json
python scripts/train_vae.py -c config/vae_both.json

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 scripts/train_vae.py -c config/vae_edente.json -g 4
```

### Durée estimée (1 GPU RTX 3090)

| Config   | Durée   |
| -------- | ------- |
| `edente` | ~10-12h |
| `dente`  | ~10-12h |
| `both`   | ~20-24h |

______________________________________________________________________

## Options utiles

### Performance

```bash
# Caching (50% du dataset en RAM)
python scripts/train_vae.py -c config/vae_edente.json --cache-rate 0.5

# Plus de workers pour le data loading
python scripts/train_vae.py -c config/vae_edente.json --num-workers 8

# Combiner
python scripts/train_vae.py -c config/vae_edente.json --cache-rate 0.5 --num-workers 8
```

### Overrides

```bash
# Changer batch size, learning rate, epochs
python scripts/train_vae.py \
  -c config/vae_edente.json \
  --batch-size 16 \
  --lr 5e-5 \
  --max-epochs 150
```

### Reprise après interruption

Modifier le config JSON :

```json
{
  "resume_ckpt": true,
  "checkpoint_dir": "./runs/vae_edente/trained_weights/checkpoint_epoch73.pth"
}
```

______________________________________________________________________

## Monitoring

### Weights & Biases

Créer un fichier `.env` :

```bash
WANDB_PROJECT=pti-ldm-vae
WANDB_ENTITY=votre-username
```

Métriques disponibles : `train/recon_loss`, `val/recon_loss`, triplets, etc.

### Validation samples

Images sauvegardées tous les 5 epochs (à partir de epoch 10) :

```
runs/vae_edente/validation_samples/epoch_15/
├── originale/
├── reconstruction/
└── diff/
```

### Critères de qualité

✅ **`val/recon_loss` < 0.05** (diminue progressivement)
✅ **Reconstructions fidèles** (pas de flou excessif)
✅ **Images diff/ sombres** (faible différence)
❌ **Éviter** : reconstructions identiques (collapse) ou noires

______________________________________________________________________

## Troubleshooting

| Problème                   | Cause                    | Solution                        |
| -------------------------- | ------------------------ | ------------------------------- |
| **Reconstructions floues** | `kl_weight` trop élevé   | Réduire à `5e-7` dans le config |
| **Loss NaN**               | Learning rate trop élevé | `--lr 1e-5`                     |
| **Out of Memory**          | Batch size trop grand    | `--batch-size 4`                |

______________________________________________________________________

## Après l'entraînement

### Inférence

```bash
python scripts/inference_vae.py \
  --checkpoint runs/vae_edente/trained_weights/autoencoder_epoch73.pth \
  --input-dir data/edente/ \
  --num-samples 20
```

### Analyse de l'espace latent

```bash
# UMAP
python scripts/analyze_static.py \
  --vae-weights runs/vae_both/trained_weights/autoencoder_epoch73.pth \
  --config-file config/vae_both.json \
  --folder-edente data/edente/ \
  --folder-dente data/dente/ \
  --output-dir results/umap \
  --method umap
```

______________________________________________________________________

## Résumé rapide

```bash
# 1. Test rapide
python scripts/train_vae.py -c config/vae_config.json --subset-size 100 --max-epochs 5

# 2. Créer vae_edente.json, vae_dente.json, vae_both.json (copier/modifier vae_config.json)

# 3. Entraînement complet
python scripts/train_vae.py -c config/vae_edente.json
python scripts/train_vae.py -c config/vae_dente.json
python scripts/train_vae.py -c config/vae_both.json

# 4. Inférence
python scripts/inference_vae.py \
  --checkpoint runs/vae_edente/trained_weights/autoencoder_epoch73.pth \
  --input-dir data/edente/
```
