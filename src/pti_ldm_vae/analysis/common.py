import random
from pathlib import Path

import numpy as np
import torch
from monai.config import print_config

from monai.transforms import Compose

from pti_ldm_vae.analysis import LatentSpaceAnalyzer, load_image_paths
from pti_ldm_vae.data.dataloaders import build_vae_preprocess_transform
from pti_ldm_vae.models import VAEModel
from pti_ldm_vae.utils.vae_loader import load_vae_config as load_vae_config_util
from pti_ldm_vae.utils.vae_loader import load_vae_model as load_vae_model_util


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device_and_output(output_dir: str) -> tuple[torch.device, Path]:
    """Setup compute device and output directory.

    Args:
        output_dir: Path to output directory

    Returns:
        Tuple of (device, output_dir_path)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print_config()
    print(f"Device: {device}")
    print(f"Output directory: {output_path}")

    return device, output_path


def load_vae_model(config_file: str, vae_weights: str, device: torch.device) -> VAEModel:
    """Load VAE model from config and weights using shared loaders.

    Args:
        config_file: Path to config JSON file.
        vae_weights: Path to VAE weights file.
        device: Torch device to load model on.

    Returns:
        Loaded VAE model in eval mode.
    """
    config = load_vae_config_util(config_file)
    vae = load_vae_model_util(config, vae_weights, device)
    vae.eval()
    print(f"Loaded VAE from {vae_weights}")
    return vae


def create_transforms(patch_size: tuple[int, int]) -> Compose:
    """Create MONAI transforms pipeline for image preprocessing."""
    return build_vae_preprocess_transform(patch_size, use_tif_reader=True)


def encode_single_image(analyzer: LatentSpaceAnalyzer, image_path: str) -> tuple[np.ndarray, str]:
    """Encode a single image to latent space.

    Args:
        analyzer: LatentSpaceAnalyzer instance
        image_path: Path to image file

    Returns:
        Tuple of (latent_vector, patient_id)
    """
    latent, ids = analyzer.encode_images([image_path])
    return latent[0], ids[0]


def collect_image_paths(folder_path: str, max_images: int) -> list[str]:
    """Collect image paths from a folder.

    Args:
        folder_path: Path to image folder
        max_images: Maximum number of images to collect

    Returns:
        List of image paths
    """
    return load_image_paths(folder_path, max_images)


def load_and_encode_group(
    analyzer: LatentSpaceAnalyzer, folder_path: str, max_images: int, group_name: str
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load and encode a group of images.

    Args:
        analyzer: LatentSpaceAnalyzer instance
        folder_path: Path to image folder
        max_images: Maximum number of images to load
        group_name: Name of the group (for logging)

    Returns:
        Tuple of (latent_vectors, image_ids, image_paths)
    """
    print(f"\nLoading {group_name} images from {folder_path}...")
    paths = load_image_paths(folder_path, max_images)
    print(f"Found {len(paths)} images")

    print(f"Encoding {group_name}...")
    latent, ids = analyzer.encode_images(paths)
    print(f"Encoded {len(latent)} images to latent space")

    return latent, ids, paths


def load_and_encode_group_with_cache(
    analyzer: LatentSpaceAnalyzer,
    folder_path: str,
    vae_weights: str,
    max_images: int,
    patch_size: tuple[int, int],
    group_name: str,
    cache_dir: Path = Path("cache/latents"),
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load and encode a group of images with intelligent caching.

    This function uses per-image caching to avoid re-encoding images that
    have already been processed. Cache is organized by VAE model signature.

    Args:
        analyzer: LatentSpaceAnalyzer instance
        folder_path: Path to image folder
        vae_weights: Path to VAE weights file (for cache key)
        max_images: Maximum number of images to load
        patch_size: Image patch size (H, W) for cache key
        group_name: Name of the group (for logging)
        cache_dir: Root directory for cache (default: cache/latents)

    Returns:
        Tuple of (latent_vectors, image_ids, image_paths)
    """
    from pti_ldm_vae.analysis.latent_cache import LatentCache

    image_paths = collect_image_paths(folder_path, max_images)

    def encoder_fn(img_path: str) -> tuple[np.ndarray, str]:
        return encode_single_image(analyzer, img_path)

    cache = LatentCache(cache_root=cache_dir)
    latents, ids, paths = cache.get_or_encode_batch(
        image_paths=image_paths,
        encoder_fn=encoder_fn,
        vae_weights=vae_weights,
        patch_size=patch_size,
        group_name=group_name,
    )

    return latents, ids, paths


def save_visualization_and_legend(
    analyzer: LatentSpaceAnalyzer,
    projections: list[tuple],
    output_dir: Path,
    title: str,
    color_by_patient: bool,
    ids_group1: list[str],
    ids_group2: list[str] | None,
    plot_filename: str,
    image_paths_list: list[list[str]] | None = None,
) -> None:
    """Save visualization plot and color legend if needed.

    Args:
        analyzer: LatentSpaceAnalyzer instance
        projections: List of (projection, ids, marker, name) tuples
        output_dir: Output directory path
        title: Plot title
        color_by_patient: Whether coloring by patient ID
        ids_group1: Image IDs for group 1
        ids_group2: Image IDs for group 2 (optional)
        plot_filename: Output plot filename (e.g., "umap_projection.html")
        image_paths_list: Optional list of image paths for each group
    """
    print("\n📊 Creating visualizations...")

    save_path = output_dir / plot_filename
    analyzer.plot_projection_2d(
        projections=projections,
        output_path=str(save_path),
        title=title,
        color_by_patient=color_by_patient,
        show_labels=color_by_patient,
        image_paths_list=image_paths_list,
    )
    print(f"✅ Plot saved to {save_path}")

    if color_by_patient:
        all_ids = ids_group1
        if ids_group2:
            all_ids = all_ids + ids_group2
        patient_to_id, patient_to_color = analyzer.create_patient_colormap(all_ids)
        legend_path = output_dir / "color_legend.txt"
        analyzer.save_color_legend(patient_to_id, patient_to_color, legend_path)
        print(f"✅ Color legend saved to {legend_path}")


def compute_and_save_statistics(
    analyzer: LatentSpaceAnalyzer,
    projection_group1: np.ndarray,
    projection_group2: np.ndarray,
    latent_group1: np.ndarray,
    latent_group2: np.ndarray,
    ids_group1: list[str],
    ids_group2: list[str],
    group1_name: str,
    group2_name: str,
    output_dir: Path,
) -> None:
    """Compute and save group statistics.

    Args:
        analyzer: LatentSpaceAnalyzer instance
        projection_group1: 2D projection for group 1
        projection_group2: 2D projection for group 2
        latent_group1: Latent vectors for group 1
        latent_group2: Latent vectors for group 2
        ids_group1: Image IDs for group 1
        ids_group2: Image IDs for group 2
        group1_name: Name of group 1
        group2_name: Name of group 2
        output_dir: Output directory path
    """
    print("\n📈 Computing group statistics...")

    projection_data = [(projection_group1, ids_group1, group1_name), (projection_group2, ids_group2, group2_name)]
    latent_data = [(latent_group1, ids_group1, group1_name), (latent_group2, ids_group2, group2_name)]

    analyzer.compute_group_statistics(projection_data, latent_data, output_dir)
    print(f"✅ Statistics saved to {output_dir}/distance_metrics.txt")
    print(f"✅ Sorted exams saved to {output_dir}/exams_sorted_by_distance.txt")
