# Analysis Module

This module provides tools for analyzing trained VAE models and evaluating generated images.

## Components

### `metrics.py`

Comprehensive image comparison and metrics computation for medical image analysis.

**Key Features:**

- Reconstruction metrics: MSE, SSIM, PSNR
- Segmentation metrics: Dice coefficient, IoU
- Perceptual metrics: VGG16 feature similarity
- Morphological metrics: Height and width measurements

**Main Class:** `ImageComparison`

```python
from pti_ldm_vae.analysis import ImageComparison

# Initialize
comparer = ImageComparison(apply_otsu_mask=True)

# Process all images
metrics_avg, metrics_ci95 = comparer.process_all_images(
    folder_paths=["./validation_samples/epoch_50/edente"],
    file_selection_mode="all",
    verbose=True,
    save_csv=True
)
```

**Key Methods:**

- `get_image_pair()` - Load ground truth and prediction pairs
- `compare_images_and_display_metrics()` - Compute all metrics
- `process_all_images()` - Batch processing with statistics
- `plot_metric_distributions_with_ci()` - Visualization

### `latent_space.py`

Tools for analyzing and visualizing VAE latent space using dimensionality reduction.

**Key Features:**

- Image encoding to latent space
- UMAP dimensionality reduction
- t-SNE dimensionality reduction
- Distance metrics between groups
- 2D projection visualization

**Main Class:** `LatentSpaceAnalyzer`

```python
from pti_ldm_vae.analysis import LatentSpaceAnalyzer
from pti_ldm_vae.models import VAEModel

# Initialize
vae = VAEModel.from_config(config).to(device)
vae.load_state_dict(torch.load("vae_weights.pth"))
analyzer = LatentSpaceAnalyzer(vae, device, transforms)

# Encode images
latent_vectors, image_ids = analyzer.encode_images(image_paths)

# UMAP projection
umap_projection, umap_model = analyzer.reduce_dimensionality_umap(
    latent_vectors,
    n_neighbors=40,
    min_dist=0.5
)

# Visualize
analyzer.plot_projection_2d(
    projections=[(umap_projection, image_ids, 'o', "group1")],
    output_path="umap_visualization.html",
    title="UMAP Projection",
    color_by_patient=True
)
```

**Key Methods:**

- `encode_images()` - Encode images to latent space
- `reduce_dimensionality_umap()` - UMAP reduction
- `reduce_dimensionality_tsne()` - t-SNE reduction
- `plot_projection_2d()` - Create 2D visualizations
- `compute_group_statistics()` - Distance metrics between groups

**Helper Functions:**

- `load_image_paths()` - Load image paths from directory
- `extract_patient_id_from_filename()` - Parse patient IDs from filenames
- `compute_distance_metrics()` - Compute distance between point clouds

## Usage Examples

### Metric Computation

```python
from pti_ldm_vae.analysis import ImageComparison

# Create comparer
comparer = ImageComparison()

# Compute metrics for validation results
metrics_avg, metrics_ci95 = comparer.process_all_images(
    folder_paths=["./validation_samples/epoch_50/edente"],
    file_selection_mode="all",
    verbose=True,
    save_csv=True
)

print(f"Average SSIM: {metrics_avg['SSIM']:.4f}")
print(f"Average Dice: {metrics_avg['Dice Coefficient']:.4f}")
```

### Latent Space Analysis

```python
from pti_ldm_vae.analysis import LatentSpaceAnalyzer, load_image_paths
from pti_ldm_vae.models import VAEModel
import torch

# Setup
device = torch.device("cuda")
vae = VAEModel.from_config(config).to(device)
vae.load_state_dict(torch.load("vae_weights.pth"))

# Load images
paths = load_image_paths("./data/edente", max_images=1000)

# Create analyzer
analyzer = LatentSpaceAnalyzer(vae, device, transforms)

# Encode and reduce
latent, ids = analyzer.encode_images(paths)
umap_proj, _ = analyzer.reduce_dimensionality_umap(latent)

# Visualize
analyzer.plot_projection_2d(
    projections=[(umap_proj, ids, 'o', "edente")],
    output_path="latent_space.html",
    title="VAE Latent Space",
    color_by_patient=True
)
```

### Two-Group Comparison

```python
# Encode two groups
latent1, ids1 = analyzer.encode_images(paths_edente)
latent2, ids2 = analyzer.encode_images(paths_dente)

# UMAP on first group
umap1, model = analyzer.reduce_dimensionality_umap(latent1)

# Transform second group
from sklearn.decomposition import PCA
pca = PCA(n_components=50).fit(latent1)
latent2_pca = pca.transform(latent2)
umap2 = model.transform(latent2_pca)

# Visualize both
analyzer.plot_projection_2d(
    projections=[
        (umap1, ids1, 'o', "edentulous"),
        (umap2, ids2, '^', "dental")
    ],
    output_path="comparison.html",
    title="Edentulous (o) vs Dental (^)",
    color_by_patient=True
)

# Compute statistics
analyzer.compute_group_statistics(
    projections=[(umap1, ids1, "edente"), (umap2, ids2, "dente")],
    latent_vectors_list=[(latent1, ids1, "edente"), (latent2, ids2, "dente")],
    output_dir=Path("./analysis_results")
)
```

## Output Files

### Metrics (`process_all_images`)

- `_metrics.csv` - Average metrics with confidence intervals
- `_dimensions.csv` - Object dimensions per image
- `_metrics_distribution.png` - Distribution plots

### Latent Space Analysis

- `umap_projection.html` / `tsne_projection.html` - Interactive 2D visualization
- `color_legend.txt` - Color mapping for patients
- `distance_metrics.txt` - Per-patient distance statistics
- `exams_sorted_by_distance.txt` - Sorted patient list

## Notes

- Image filenames should follow: `<slice_id>_<date>_<patient_id>.tif`
- Example: `1000_HA_2021_02_545.tif` (patient ID is 545)
- Patient IDs (last element after underscore split) are used for grouping
- Metrics module expects paired folders: `edente/` and `edente_synth/`
- Analysis scripts use PCA preprocessing (50 components) before UMAP/t-SNE
- All tools support both CPU and CUDA execution
