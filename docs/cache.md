# Cache

This page documents the latent analysis cache (disk).

## 1. Latent analysis cache (disk)

Used by:

- `python -m pti_ldm_vae_v2.analysis.analyze_interactive`
- `python -m pti_ldm_vae_v2.analysis.analyze_static`

Location:

```
<run_dir>/analysis/latents_cache/
```

How it works:

- Each image gets a cached `.npz` file (latent + patient_id).
- Cache signature includes:
  - VAE checkpoint path
  - `patch_size`
  - image modification time
- If an input image changes, it is re-encoded automatically.

## 2. Clearing the cache

You can clear the cache by removing the related folder:

```
rm -rf <run_dir>/analysis/latents_cache/
```

> [!NOTE]
> The latent cache is only used by analysis tools, not by training or sampling.
