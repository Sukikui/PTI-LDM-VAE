# Tools

Small utilities used to prepare datasets or compute auxiliary artifacts (not ML models).

## `compute_mask_metrics.py`

Compute mask-based attributes used as targets:

- AR-VAE: `regularized_attributes.attribute_file`
- Regression head: `data.attributes_path` / `evaluation.attributes_path`

The script reads paired mask TIFFs from `--edente-dir` and `--dente-dir` (same filenames), then writes:

- `attributes_edente.json`: `{ "<filename>.tif": { "height_0": int, "width_0": int, ... } }`
- `attributes_dente.json`: `{ "<filename>.tif": { "height_0": int, "width_0": int, ... } }`

### Usage

```bash
python -m pti_ldm_vae_v2.tools.compute_mask_metrics \
  --edente-dir data/edente \
  --dente-dir data/dente \
  --output-edente data/metrics/attributes_edente.json \
  --output-dente data/metrics/attributes_dente.json
```

