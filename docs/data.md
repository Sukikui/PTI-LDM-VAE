# Data

This project uses `.tif` images organized by source (`dente`, `edente`) and optional metric JSON files.
The folder layout is fixed as below.

## 1. Layout

```
data/
|-- train_val/
|   |-- dente/
|   |   |-- 339_HM_2014_11_2691.tif
|   |   `-- 340_HM_2014_11_2692.tif
|   |-- edente/
|   |   |-- 339_HM_2014_11_2691.tif
|   |   `-- 340_HM_2014_11_2692.tif
|   `-- metrics/
|       |-- attributes_edente.json
|       `-- attributes_dente.json       (optional)
`-- test/
    |-- dente/
    |   |-- 101_HA_2020_05_1001.tif     (different patients from train_val/)
    |   `-- 102_HA_2020_05_1002.tif
    |-- edente/
    |   |-- 101_HA_2020_05_1001.tif
    |   `-- 102_HA_2020_05_1002.tif
    `-- metrics/
        |-- attributes_edente.json
        `-- attributes_dente.json       (optional)
```

## 2. File naming and pairing rules

There are two different behaviors depending on the module.

**LDM (paired data required)**

- The LDM expects **paired** images: one dentate + one edentulous.
- Files are paired by **sorted filename order**.
- The two folders must contain the **same filenames** and **same count**.

**VAE + regression head (no pairing required)**

- If `data_source="edente"` or `"dente"`, only that folder is used.
- If `data_source="both"`, the two folders are concatenated (not paired).

## 3. Dataset patient splitting

The **test set must not contain any images from patients present in train/val**.
Do not mix patient IDs between `data/train_val` and `data/test`, even if the slices differ.
Slices from the same patient are highly similar and will bias evaluation.

## 4. Image format

All training and inference pipelines expect:

- `.tif` files (not `.tiff` for training/inference).
- Single-channel images.
- Background pixels should be zero, because normalization uses a non-zero mask.

## 5. Metrics / attributes JSON

Metrics are stored as a JSON mapping from filename to values:

```json
{
  "339_HM_2014_11_2691.tif": {
    "height_0": 123,
    "width_0": 45,
    "width_1": 50
  }
}
```

Where this is used:

- **Regression head**: `targets` defines which keys are required and their order.
- **AR-VAE**: `regularized_attributes.attribute_latent_mapping` defines the required keys.

Metrics can be generated with:

```
python -m pti_ldm_vae_v2.tools.compute_mask_metrics
```

Quickly preview a single `.tif` (keeps background black):

```
python -m pti_ldm_vae_v2.tools.view_tif --input-path data/test/dente/example.tif
```
