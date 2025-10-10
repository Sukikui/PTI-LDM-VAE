# PTI-LDM-VAE

Pipeline complet d'entraînement et d'inférence pour la génération d'images médicales TIF à l'aide d'un Variational Autoencoder (VAE) couplé à un Latent Diffusion Model (LDM) conditionné.

## Description

Ce projet implémente un pipeline en deux étapes pour la génération d'images médicales :

1. **VAE (Variational Autoencoder)** : Entraînement d'un auto-encodeur variationnel sur des images TIF float32 (édentée → édentée) pour apprendre une représentation latente compacte
2. **LDM (Latent Diffusion Model)** : Entraînement d'un modèle de diffusion conditionné (dentée → édentée) qui opère dans l'espace latent du VAE

Les images sont des fichiers TIF float32 mono-canal (256×256). Le LDM est conditionné par l'image dentée encodée par le VAE et génère une image édentée correspondante.

**Auteur original** : Tuong Vy PHAM (tv.pham1996@gmail.com)

## Fonctionnalités

- Entraînement VAE avec discriminateur adversarial (PatchGAN)
- Entraînement LDM conditionné avec cross-attention
- Inférence VAE et LDM
- Calcul de métriques (PSNR, SSIM, Dice, IoU, métriques géométriques)
- Visualisation de l'espace latent (UMAP, t-SNE)
- Support multi-GPU avec Distributed Data Parallel (DDP)
- Suivi d'entraînement avec TensorBoard
- Sauvegarde automatique des meilleurs checkpoints

## Structure du projet

```
PTI-LDM-VAE/
├── train_autoencoder_tif.py          # Entraînement du VAE
├── train_diffusion_tif_cond.py       # Entraînement du LDM conditionné
├── inference_vae_tif.py              # Inférence VAE
├── compute_metrics_class_tif.py      # Calcul des métriques de qualité
├── umap_latent_vae.py                # Visualisation UMAP de l'espace latent
├── tsne_latent_vae.py                # Visualisation t-SNE de l'espace latent
├── visualize_image.py                # Utilitaires de visualisation
├── utils_tif.py                      # Dataset, augmentations et I/O TIF
├── utils_tif_no_augment.py           # Variante sans augmentations
├── requirements.txt                  # Dépendances pip
├── environment.yml                   # Environnement conda
└── config/                           # Fichiers de configuration (non inclus)
    ├── config_train_16g_cond.json    # Config LDM
    └── environment_tif.json          # Chemins et paramètres
```

## Installation

### Prérequis

- Python 3.10
- CUDA compatible GPU (recommandé)
- Conda ou venv

### Installation avec Conda (recommandé)

```bash
# Créer l'environnement conda
conda env create -f environment.yml

# Activer l'environnement
conda activate pti-ldm-vae
```

### Installation avec pip

```bash
# Créer un environnement virtuel
python -m venv .venv

# Activer l'environnement
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Installation de PyTorch

Installer PyTorch selon votre configuration (OS, CUDA) :
```bash
# Exemple pour CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Voir les instructions officielles : https://pytorch.org/get-started/locally/

## Configuration

Le projet utilise des fichiers JSON pour la configuration :

- `config/environment_tif.json` : Chemins des datasets, poids, options d'I/O
- `config/config_train_16g_cond.json` : Hyperparamètres du LDM (architecture, learning rate, epochs, etc.)

**Note** : Les fichiers de configuration ne sont pas inclus dans ce dépôt. Créez-les selon vos besoins ou contactez l'auteur.

## Utilisation

### 1. Entraînement du VAE

Entraîne un auto-encodeur variationnel sur des images édentées :

```bash
python train_autoencoder_tif.py \
    --environment-file ./config/environment_tif.json \
    --config-file ./config/config_train_16g_cond.json \
    --gpus 1
```

**Sorties** :
- `trained_weights/autoencoder/` : Poids du modèle (`.pth`)
- `tfevent/` : Logs TensorBoard
- `validation_samples/` : Échantillons de validation

**Options** :
- `--gpus N` : Nombre de GPUs (active DDP si > 1)

### 2. Entraînement du LDM conditionné

Entraîne le modèle de diffusion conditionné (nécessite un VAE pré-entraîné) :

```bash
python train_diffusion_tif_cond.py \
    --environment-file ./config/environment_tif.json \
    --config-file ./config/config_train_16g_cond.json \
    --gpus 1
```

Le script charge automatiquement le VAE pré-entraîné spécifié dans `environment_tif.json`.

**Sorties par epoch** :
- `validation_samples/epoch_N/edente/` : Ground truth
- `validation_samples/epoch_N/edente_synth/` : Images générées
- `trained_weights/diffusion_unet_epochN.pth` : Meilleurs poids
- `trained_weights/checkpoint_epochN.pth` : Checkpoint complet

### 3. Inférence VAE

Lance l'inférence sur de nouvelles images :

```bash
python inference_vae_tif.py
```

**À configurer dans le script** (lignes 30-42) :
- `weights_path` : Chemin vers le checkpoint du VAE
- `input_dir` : Dossier contenant les images TIF d'entrée
- `description` : Description pour le nom du dossier de sortie

**Sorties** :
- `inference_<nom>_<epoch>_<description>/résultats_tif/` : Résultats en TIF
- `inference_<nom>_<epoch>_<description>/résultats_png/` : Visualisations PNG

### 4. Calcul des métriques

Calcule les métriques de qualité entre images GT et prédites :

```bash
python compute_metrics_class_tif.py
```

**Structure attendue des dossiers** :
```
<run_dir>/validation_samples/epoch_N/
├── edente/              # Ground truth
│   ├── step000.tif
│   └── step001.tif
└── edente_synth/        # Prédictions
    ├── step000.tif
    └── step001.tif
```

**À configurer dans le script** (section `main`) :
- `folder_path` : Chemin vers le dossier d'exécution
- `num_epoch` : Numéro d'epoch à analyser

**Métriques calculées** :
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Dice coefficient
- IoU (Intersection over Union)
- Métriques géométriques (dimensions, excentricité, etc.)

**Sorties** :
- `*_metrics.csv` : Métriques par image
- `*_dimensions.csv` : Dimensions des objets
- `*_metrics_distribution.png` : Distributions des métriques

### 5. Visualisation de l'espace latent

#### UMAP

```bash
python umap_latent_vae.py
```

**À configurer dans le script** :
- `vae_weights` : Chemin vers les poids du VAE
- `folder_edentee` / `folder_dentee` : Dossiers d'images
- `max_images` : Nombre maximum d'images à encoder

#### t-SNE

```bash
python tsne_latent_vae.py
```

Configuration similaire à UMAP.

**Sorties** :
- Visualisation 2D de l'espace latent
- Fichier de légende des couleurs (si mode par examen activé)

## Monitoring avec TensorBoard

Visualiser l'entraînement en temps réel :

```bash
tensorboard --logdir=<run_dir>/tfevent
```

**Visualisations disponibles** :
- Courbes de loss (train/validation)
- Triplets d'images (original, reconstruction, différence)
- Grilles de validation avec bruitages intermédiaires
- Comparaisons condition/GT/génération

## Reprise d'entraînement

Les deux scripts d'entraînement supportent la reprise depuis un checkpoint :

```python
# Dans le fichier de config JSON
{
    "resume_ckpt": true,
    "checkpoint_dir": "path/to/checkpoint_epochN.pth",
    "start_epoch": N  # Pour le LDM uniquement
}
```

Le checkpoint contient :
- États des modèles (autoencoder, discriminator/unet)
- États des optimiseurs
- Meilleure validation loss
- Numéro d'epoch actuel
- Compteur de steps global

## Architecture

### VAE (Variational Autoencoder)

- Architecture : AutoencoderKL (MONAI)
- Loss : L1/L2 + KL divergence + Perceptual (VGG) + Adversarial
- Discriminateur : PatchDiscriminator 3 couches
- Warm-up : 5 epochs sans adversarial loss
- Latent dim : 4 canaux

### LDM (Latent Diffusion Model)

- Architecture : UNet avec cross-attention
- Scheduler : DDPM (Denoising Diffusion Probabilistic Model)
- Conditioning : Projection linéaire du latent dentée vers cross_attention_dim
- Mixed precision training (AMP)
- Scale factor : Calculé automatiquement depuis le VAE

## Dépendances principales

- PyTorch 2.5.1
- MONAI 1.5.1
- TensorFlow 2.20.0 (backend pour certaines métriques)
- albumentations 2.0.8 (augmentations)
- scikit-learn 1.7.2
- UMAP 0.1.1
- tifffile 2024.9.20
- TensorBoard 2.18.0

Voir `requirements.txt` ou `environment.yml` pour la liste complète.

## Données

### Format

- Type : TIF float32
- Canaux : 1 (niveaux de gris)
- Résolution : 256×256 pixels
- Normalisation : Centrée-réduite par masque (pixels non-nuls)

### Organisation

```
data/
├── train/
│   ├── edente/      # Images édentées (VAE)
│   └── dente/       # Images dentées (LDM condition)
└── inference/
    ├── edente/
    └── dente/
```

## Conseils d'utilisation

### Performance

- Utilisez DDP (`--gpus > 1`) pour l'entraînement multi-GPU
- Le cache des datasets peut être ajusté (`cache=1.0` pour tout en RAM)
- AMP activé par défaut pour le LDM (gain mémoire/vitesse)

### Hyperparamètres clés

- `kl_weight` : Poids de la KL divergence (crucial pour la qualité du latent)
  - Trop élevé → mauvaise reconstruction
  - Trop faible → latent non régularisé
- `perceptual_weight` : Poids de la perceptual loss
- `lr` : Learning rate (multiplié par world_size en DDP)

### Sauvegarde

- Seuls les **meilleurs** checkpoints sont conservés (validation loss minimale)
- Les anciens checkpoints sont automatiquement supprimés
- Les poids `*_last.pth` sont toujours l'epoch la plus récente

## Licence

Copyright (c) MONAI Consortium
Licensed under the Apache License, Version 2.0

## Contact

Pour toute question sur le code ou la méthodologie :
- **Tuong Vy PHAM** : tv.pham1996@gmail.com

## Références

Le code s'appuie sur :
- [MONAI](https://monai.io/) - Medical Open Network for AI
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) - Rombach et al.
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - Kingma & Welling

## Troubleshooting

### Erreur CUDA Out of Memory
- Réduire `batch_size` dans la config
- Réduire `patch_size`
- Activer gradient checkpointing (si disponible dans le modèle)

### Reconstruction floue (VAE)
- Augmenter `perceptual_weight`
- Réduire `kl_weight`
- Augmenter le nombre de canaux

### LDM ne converge pas
- Vérifier que le `scale_factor` est proche de 1
- Vérifier la qualité du VAE pré-entraîné
- Augmenter `num_train_timesteps`

### Images de validation noires
- Vérifier la normalisation des données
- Vérifier que le masque (pixels non-nuls) est correct
- Ajuster les percentiles dans `normalize_batch_for_display`