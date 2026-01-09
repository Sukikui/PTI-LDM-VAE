# Latent Diffusion Model (LDM)

This package contains a minimal, standalone LDM training and sampling pipeline that mirrors the
legacy behavior while using the v2 codebase.

## Commands

Train:

```bash
python -m pti_ldm_vae_v2.ldm.train -c config/ldm_both_no_adv.json
```

Train (metrics-only conditioning + noisy dentate init):

```bash
python -m pti_ldm_vae_v2.ldm.train -c config/ldm_both_no_adv_metrics_only_noisy.json
```

Sample (inference):

```bash
python -m pti_ldm_vae_v2.ldm.sample \
  -c config/ldm_both_no_adv.json \
  --checkpoint runs/ldm_both_no_adv/trained_weights/ldm_unet_best.pth \
  --input-dir data/test/dente
```

## Key Arguments

Training (`pti_ldm_vae_v2.ldm.train`):
- `-c/--config-file`: LDM JSON config.
- `--max-epochs`: Override epochs.
- `--batch-size`: Override batch size.
- `--lr`: Override learning rate.

Sampling (`pti_ldm_vae_v2.ldm.sample`):
- `-c/--config-file`: LDM JSON config.
- `--checkpoint`: UNet checkpoint (best or last).
- `--input-dir`: Dentate input folder.
- `--edente-dir`: Optional edentulous ground-truth folder (defaults to a sibling `edente` folder).
- `--output-dir`: Optional output override.
- `--num-steps`: DDIM steps (default 50).
- `--eta`: DDIM eta (default 0.0).
- `--guidance-scale`: Classifier-free guidance scale.
- `--drop-z`, `--drop-metrics`: Conditioning dropout during sampling.
- `--batch-size`: Inference batch size.
- `--num-samples`: Optional cap on processed images.

Configuration (`conditioning` block):
- `use_dentate_latent`: Enable/disable dentate latent conditioning (default true). When false, `concat_dentate` is forced to false.

Configuration (`noise_init` block):
- `init_mode`: ``pure_noise`` (default) or ``dentate_noisy`` to start from a noisy dentate latent.
- `noise_top`: Noise scale at the top of the latent (default 1.0).
- `noise_bottom`: Noise scale at the bottom of the latent (default 0.0).
- `noise_exponent`: Exponent shaping the vertical gradient (default 1.0).
- `noise_direction`: ``vertical`` (top->bottom) or ``horizontal`` (left->right).
- `noise_weight`: Global multiplier applied to the noise (default 1.0).

The same `noise_init` settings are used during training to build the noise input when `init_mode` is
``dentate_noisy``, so train and inference stay aligned.

Noisy latent visualization:

```bash
python -m pti_ldm_vae_v2.ldm.visualize_noisy_latent \
  -c config/ldm_both_no_adv.json \
  --input-path data/test/dente/example.tif
```
This starts a Dash app with a channel selector. It uses only the VAE encoder plus the `sampling` block.

## Outputs

Training artifacts:
- Checkpoints: `runs/<run_name>/trained_weights/ldm_unet_best.pth`, `runs/<run_name>/trained_weights/ldm_unet_last.pth`
- Splits: `runs/<run_name>/splits/ldm_pairs.json`

Sampling outputs (default):

```
runs/<run_name>/results/<input_dir_relative>/
  results_tif/
  results_png/
```

Each output uses the original dentate filename and contains triplets ordered as:
`dente (gt) | edente (gt) | edente (pred)`.
