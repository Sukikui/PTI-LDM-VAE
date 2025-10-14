"""
Latent space analysis tools for VAE models.

This module provides tools for analyzing and visualizing the latent space
of trained VAE models using dimensionality reduction techniques (UMAP, t-SNE).
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from glob import glob
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def extract_exam_id_from_filename(filename: str) -> str:
    """
    Extract exam ID from filename by removing the first part before underscore.

    Args:
        filename: Image filename (e.g., "0001_exam_ABC.tif")

    Returns:
        Exam ID (e.g., "exam_ABC.tif")
    """
    return filename.split("_", 1)[1] if "_" in filename else filename


def compute_distance_metrics(
    points1: np.ndarray,
    points2: np.ndarray
) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute distance metrics between two point clouds.

    Args:
        points1: First set of points [N1, D]
        points2: Second set of points [N2, D]

    Returns:
        Tuple of (center_distance, std1, std2, mean_cross_distance) or None if empty
    """
    if len(points1) == 0 or len(points2) == 0:
        return None

    points1 = np.array(points1)
    points2 = np.array(points2)

    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    center_distance = np.linalg.norm(mean1 - mean2)

    std1 = np.mean(np.std(points1, axis=0)) if len(points1) > 1 else 0.0
    std2 = np.mean(np.std(points2, axis=0)) if len(points2) > 1 else 0.0

    all_distances = cdist(points1, points2)
    mean_cross_distance = np.mean(all_distances)

    return center_distance, std1, std2, mean_cross_distance


class LatentSpaceAnalyzer:
    """
    Analyzer for VAE latent space using dimensionality reduction.

    This class provides methods to:
    - Encode images to latent space
    - Reduce dimensionality with PCA + UMAP/t-SNE
    - Visualize latent space projections
    - Compute distance metrics between groups

    Args:
        vae_model: Trained VAE model with encode_stage_2_inputs method
        device: Torch device for computation
        transform: MONAI transform pipeline for image preprocessing
    """

    def __init__(
        self,
        vae_model: torch.nn.Module,
        device: torch.device,
        transform
    ):
        self.vae = vae_model
        self.device = device
        self.transform = transform
        self.vae.eval()

    def encode_images(
        self,
        image_paths: List[str],
        max_images: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode images to latent space.

        Args:
            image_paths: List of image file paths
            max_images: Maximum number of images to encode

        Returns:
            Tuple of (latent_vectors [N, D], image_ids)
        """
        if max_images is not None:
            image_paths = image_paths[:max_images]

        latent_vectors = []
        image_ids = []

        with torch.no_grad():
            for path in image_paths:
                img = self.transform(path).unsqueeze(0).to(self.device)
                z = self.vae.encode_stage_2_inputs(img).cpu().flatten(start_dim=1)
                latent_vectors.append(z)

                filename = os.path.basename(path)
                image_ids.append(extract_exam_id_from_filename(filename))

        return torch.cat(latent_vectors, dim=0).numpy(), image_ids

    def reduce_dimensionality_umap(
        self,
        latent_vectors: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 40,
        min_dist: float = 0.5,
        random_state: int = 42,
        pca_components: int = 50
    ) -> Tuple[np.ndarray, object]:
        """
        Reduce dimensionality using PCA + UMAP.

        Args:
            latent_vectors: Input latent vectors [N, D]
            n_components: Number of UMAP components
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            random_state: Random seed
            pca_components: Number of PCA components before UMAP

        Returns:
            Tuple of (reduced_vectors [N, n_components], umap_model)
        """
        try:
            import umap
        except ImportError:
            raise ImportError("Please install umap-learn: pip install umap-learn")

        # PCA preprocessing
        pca = PCA(n_components=pca_components)
        vectors_pca = pca.fit_transform(latent_vectors)

        # UMAP
        umap_model = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
        vectors_umap = umap_model.fit_transform(vectors_pca)

        return vectors_umap, umap_model

    def reduce_dimensionality_tsne(
        self,
        latent_vectors: np.ndarray,
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42,
        pca_components: int = 50
    ) -> np.ndarray:
        """
        Reduce dimensionality using PCA + t-SNE.

        Args:
            latent_vectors: Input latent vectors [N, D]
            n_components: Number of t-SNE components
            perplexity: Perplexity parameter for t-SNE
            random_state: Random seed
            pca_components: Number of PCA components before t-SNE

        Returns:
            Reduced vectors [N, n_components]
        """
        # PCA preprocessing
        pca = PCA(n_components=pca_components)
        vectors_pca = pca.fit_transform(latent_vectors)

        # t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init='pca',
            random_state=random_state
        )
        vectors_tsne = tsne.fit_transform(vectors_pca)

        return vectors_tsne

    def create_exam_colormap(
        self,
        exam_ids: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, tuple]]:
        """
        Create color mapping for exams.

        Args:
            exam_ids: List of exam identifiers

        Returns:
            Tuple of (exam_to_id, exam_to_color) dictionaries
        """
        unique_exams = sorted(set(exam_ids))
        exam_to_id = {exam: i for i, exam in enumerate(unique_exams)}

        colormap = cm.get_cmap('tab20', len(unique_exams))
        exam_to_color = {exam: colormap(i) for i, exam in enumerate(unique_exams)}

        return exam_to_id, exam_to_color

    def plot_projection_2d(
        self,
        projections: List[Tuple[np.ndarray, List[str], str, str]],
        output_path: str,
        title: str = "Latent Space Projection",
        color_by_exam: bool = True,
        show_labels: bool = True
    ):
        """
        Plot 2D projection of latent space.

        Args:
            projections: List of (vectors, ids, marker, name) tuples
            output_path: Path to save figure
            title: Plot title
            color_by_exam: If True, color by exam; if False, color by group
            show_labels: If True, show exam ID labels on plot
        """
        plt.figure(figsize=(10, 8))

        # Create colormap if needed
        if color_by_exam:
            all_ids = []
            for vectors, ids, _, _ in projections:
                all_ids.extend(ids)
            exam_to_id, exam_to_color = self.create_exam_colormap(all_ids)

        # Plot each projection
        for vectors, ids, marker, name in projections:
            if color_by_exam:
                colors = [exam_to_color[exam_id] for exam_id in ids]
            else:
                colors = name  # Use name as color string

            for i, (x, y) in enumerate(vectors):
                if color_by_exam:
                    color = colors[i]
                else:
                    color = colors
                plt.scatter(x, y, s=30, color=color, marker=marker, alpha=0.6)

        # Add labels if requested
        if show_labels and color_by_exam:
            for exam, idx in exam_to_id.items():
                # Add labels for each group
                for vectors, ids, marker, name in projections:
                    exam_points = [vectors[i] for i in range(len(ids)) if ids[i] == exam]
                    if exam_points:
                        mean_point = np.mean(exam_points, axis=0)
                        marker_symbol = 'o' if marker == 'o' else '△'
                        plt.text(
                            mean_point[0], mean_point[1], f"{marker_symbol}{idx}",
                            fontsize=9, weight='bold',
                            ha='center', va='center', color='black', alpha=0.9
                        )

        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def compute_group_statistics(
        self,
        projections: List[Tuple[np.ndarray, List[str], str]],
        latent_vectors_list: List[Tuple[np.ndarray, List[str], str]],
        output_dir: Path
    ):
        """
        Compute and save statistics for grouped data.

        Args:
            projections: List of (reduced_vectors, ids, name) tuples
            latent_vectors_list: List of (latent_vectors, ids, name) tuples
            output_dir: Directory to save results
        """
        if len(projections) != 2 or len(latent_vectors_list) != 2:
            return

        proj1, ids1, name1 = projections[0]
        proj2, ids2, name2 = projections[1]
        lat1, _, _ = latent_vectors_list[0]
        lat2, _, _ = latent_vectors_list[1]

        # Group by exam
        exam_data_proj = defaultdict(lambda: {name1: [], name2: []})
        exam_data_lat = defaultdict(lambda: {name1: [], name2: []})

        for i, exam_id in enumerate(ids1):
            exam_data_proj[exam_id][name1].append(proj1[i])
            exam_data_lat[exam_id][name1].append(lat1[i])

        for i, exam_id in enumerate(ids2):
            exam_data_proj[exam_id][name2].append(proj2[i])
            exam_data_lat[exam_id][name2].append(lat2[i])

        # Compute metrics
        results_file = output_dir / "distance_metrics.txt"
        with open(results_file, "w") as f:
            f.write("Distance Metrics per Exam (Latent Space and Projection)\n")
            f.write("=" * 60 + "\n\n")

            exam_distances = []
            for exam in sorted(exam_data_lat.keys()):
                lat_group1 = exam_data_lat[exam][name1]
                lat_group2 = exam_data_lat[exam][name2]
                proj_group1 = exam_data_proj[exam][name1]
                proj_group2 = exam_data_proj[exam][name2]

                if len(lat_group1) == 0 or len(lat_group2) == 0:
                    continue

                metrics_lat = compute_distance_metrics(lat_group1, lat_group2)
                metrics_proj = compute_distance_metrics(proj_group1, proj_group2)

                if not metrics_lat or not metrics_proj:
                    continue

                f.write(f"{exam}\n")
                f.write(f"  - n_{name1}: {len(lat_group1)}, n_{name2}: {len(lat_group2)}\n")
                f.write(f"  - [Latent] center_dist: {metrics_lat[0]:.3f}, "
                       f"std_{name1}: {metrics_lat[1]:.3f}, std_{name2}: {metrics_lat[2]:.3f}, "
                       f"mean_cross_dist: {metrics_lat[3]:.3f}\n")
                f.write(f"  - [Projection] center_dist: {metrics_proj[0]:.3f}, "
                       f"std_{name1}: {metrics_proj[1]:.3f}, std_{name2}: {metrics_proj[2]:.3f}, "
                       f"mean_cross_dist: {metrics_proj[3]:.3f}\n\n")

                exam_distances.append((exam, metrics_lat[0]))

        # Sort by distance
        exam_distances.sort(key=lambda x: x[1])
        sorted_file = output_dir / "exams_sorted_by_distance.txt"
        with open(sorted_file, "w") as f:
            f.write("Exams sorted by latent space center distance\n")
            f.write("=" * 60 + "\n\n")
            for exam, dist in exam_distances:
                f.write(f"{exam}: {dist:.3f}\n")

    def save_color_legend(
        self,
        exam_to_id: Dict[str, int],
        exam_to_color: Dict[str, tuple],
        output_path: Path
    ):
        """
        Save color legend to file.

        Args:
            exam_to_id: Mapping from exam ID to integer
            exam_to_color: Mapping from exam ID to color tuple
            output_path: Path to save legend
        """
        with open(output_path, "w") as f:
            f.write("Color Legend for Exams\n")
            f.write("=" * 60 + "\n\n")
            for exam in sorted(exam_to_id.keys(), key=lambda x: exam_to_id[x]):
                color_hex = mcolors.to_hex(exam_to_color[exam])
                idx = exam_to_id[exam]
                f.write(f"{idx}: {exam} — {color_hex}\n")


def load_image_paths(
    data_dir: str,
    max_images: Optional[int] = None
) -> List[str]:
    """
    Load image paths from directory.

    Args:
        data_dir: Directory containing .tif images
        max_images: Maximum number of images to load

    Returns:
        List of image paths
    """
    paths = sorted(glob(os.path.join(data_dir, "*.tif")))
    if max_images is not None:
        paths = paths[:max_images]
    return paths