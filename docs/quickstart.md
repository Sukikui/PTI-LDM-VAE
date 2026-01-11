# Quickstart

This page gives the shortest reliable path to train the full pipeline and run one LDM sampling pass.
Before starting, make sure your data matches [`docs/data.md`](data.md).

## 1. Setup the environment

Follow [`docs/setup.md`](setup.md).

## 2. Train the VAE

```bash
python -m pti_ldm_vae_v2.vae.train -c config/vae_both_no_adv.json
```

> [!NOTE]
> Checkpoints are written to `runs/vae_both_no_adv/trained_weights/`.
> You can use either `autoencoder_epoch*.pth` or `checkpoint_epoch*.pth` for downstream steps.

## 3. Train the regression head

This config expects the VAE checkpoint path inside the JSON. Update it if needed:
`config/nreg_edente_from_both.json` → `vae.checkpoint`.

```bash
python -m pti_ldm_vae_v2.regression_head.train -c config/nreg_edente_from_both.json
```

> [!NOTE]
> Checkpoints are written to `runs/nreg_edente_from_both/trained_weights/`.

## 4. Train the LDM

This config expects both the VAE and regression head checkpoints:
`config/ldm_both_no_adv.json` → `vae.checkpoint` and `regressor.checkpoint`.

```bash
python -m pti_ldm_vae_v2.ldm.train -c config/ldm_both_no_adv.json
```

> [!NOTE]
> Checkpoints are written to `runs/ldm_both_no_adv/trained_weights/`.

## 5. Sample with the LDM

```bash
python -m pti_ldm_vae_v2.ldm.sample \
  -c config/ldm_both_no_adv.json \
  --checkpoint runs/ldm_both_no_adv/trained_weights/ldm_unet_best.pth \
  --input-dir data/test/dente \
  --num-steps 1000
```

> [!NOTE]
> The sampler looks for `data/test/edente` by default. Use `--edente-dir` if your path differs.
> Results are stored under `runs/ldm_both_no_adv/results/<input_dir_relative>/`.

## Optional: quick smoke test

If you just want to validate the pipeline, run fewer epochs:

```bash
python -m pti_ldm_vae_v2.vae.train -c config/vae_both_no_adv.json --max-epochs 1
python -m pti_ldm_vae_v2.regression_head.train -c config/nreg_edente_from_both.json --max-epochs 1
python -m pti_ldm_vae_v2.ldm.train -c config/ldm_both_no_adv.json --max-epochs 1
```

For config structure and outputs, see:

- [`docs/configs.md`](configs.md)
- [`docs/runs.md`](runs.md)
