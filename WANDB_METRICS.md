# W&B Metrics (VAE)

Guide concis des métriques loggées pendant l'entraînement/validation du VAE.

## Training

- `train/recon_loss` : perte de reconstruction L1/L2 par batch (axe `train/step`), doit baisser.
- `train/kl_loss` : KL divergences par batch.
- `train/perceptual_loss` : perte perceptuelle VGG par batch.
- `train/loss_total` : somme pondérée (recon + KL * kl_weight + perceptuel + éventuel adv/AR).
- `train/ar_loss_*` : pertes AR-VAE par attribut (si activé).
- `train/triplets` : triplet (original | reconstruction | diff) loggé au premier batch de chaque époque.

## Validation

- `val/recon_loss`, `val/kl_loss`, `val/perceptual_loss`, `val/loss_total` : moyennes par époque (axe `epoch`).
- `val/ar_loss_*` : pertes AR-VAE moyennes par attribut (si activé).
- `val/triplets` : un triplet (original | reconstruction | diff) loggé périodiquement (`log_triplet_every`).

## Performance

- `epoch` : compteur d'époques (axe X pour la validation).
- `time_per_epoch` : durée d'une époque (secondes).

## Config loggée automatiquement

- Architecture : `spatial_dims`, `latent_channels`, `channels`, `num_res_blocks`, attention, etc.
- Entraînement : `batch_size`, `lr`, `max_epochs`, `kl_weight`, `perceptual_weight`, `recon_loss`, `ar_vae_weight`.
- Données : `data_source`, `augment`, `patch_size`, `train_split`, `cache_rate`.
