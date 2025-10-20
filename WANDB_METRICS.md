# Weights & Biases Metrics Guide

Guide synthétique des métriques loggées sur W&B pendant l'entraînement du VAE.

______________________________________________________________________

## 📊 Métriques d'entraînement (Training)

### `train/recon_loss`

- **Description** : Loss de reconstruction L1/L2 à chaque batch
- **Quand** : À chaque itération (step)
- **Objectif** : Doit diminuer → le modèle apprend à reconstruire les images
- **Axe X** : `train/step` (nombre de batches traités)

### `train/triplets`

- **Description** : Images (original | reconstruction | différence)
- **Quand** : Au premier batch de chaque époque
- **Utilité** : Visualiser la qualité des reconstructions pendant l'entraînement
- **Format** : 3 images côte à côte

______________________________________________________________________

## ✅ Métriques de validation (Validation)

### `val/recon_loss`

- **Description** : Loss de reconstruction moyenne sur l'ensemble de validation
- **Quand** : À la fin de chaque époque
- **Objectif** : Doit diminuer → le modèle généralise bien
- **Axe X** : `epoch`

### `val/triplet_step{XXX}`

- **Description** : Triplets (original | reconstruction | différence) pour chaque sample de validation
- **Quand** : À la fin de chaque époque
- **Utilité** : Évaluer visuellement la qualité sur les données de validation
- **Nombre** : Autant que de samples dans le dataset de validation

______________________________________________________________________

## ⏱️ Métriques de performance

### `time_per_epoch`

- **Description** : Temps d'entraînement par époque (en secondes)
- **Quand** : À la fin de chaque époque
- **Utilité** : Monitorer la vitesse d'entraînement

### `epoch`

- **Description** : Numéro de l'époque actuelle
- **Quand** : À la fin de chaque époque
- **Utilité** : Axe X pour les métriques de validation

______________________________________________________________________

## 🎯 Comment interpréter

### ✅ Bon entraînement

- `train/recon_loss` et `val/recon_loss` diminuent régulièrement
- Les courbes sont parallèles (pas d'overfitting)
- Les triplets montrent des reconstructions de plus en plus fidèles

### ⚠️ Overfitting

- `train/recon_loss` diminue mais `val/recon_loss` augmente ou stagne
- **Solution** : Augmenter la régularisation (`kl_weight`), ajouter de l'augmentation

### ⚠️ Underfitting

- Les deux loss diminuent très lentement ou stagnent tôt
- **Solution** : Augmenter la capacité du modèle (`channels`, `num_res_blocks`)

______________________________________________________________________

## 📈 Graphiques utiles dans W&B

1. **Training vs Validation Loss** : Comparer `train/recon_loss` et `val/recon_loss`
2. **Triplets Evolution** : Voir comment la qualité s'améliore
3. **Time per Epoch** : Détecter les ralentissements

______________________________________________________________________

## ⚙️ Configuration loggée automatiquement

W&B enregistre aussi tous les hyperparamètres dans l'onglet **Config** :

- Architecture (spatial_dims, latent_channels, channels, num_res_blocks)
- Training (batch_size, lr, max_epochs, kl_weight, perceptual_weight)
- Data (data_source, augment)
