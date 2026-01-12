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
You can clear the cache in two ways:

1) Delete the folder manually:
```
rm -rf <run_dir>/analysis/latents_cache/
```

2) Or call the helper:
`LatentCache.clear_cache()` in
[`src/pti_ldm_vae_v2/analysis/latent_analysis.py`](../src/pti_ldm_vae_v2/analysis/latent_analysis.py).

> [!NOTE]
> The latent cache is only used by analysis tools, not by training or sampling.
