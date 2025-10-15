"""Latent space analysis tools for VAE models.

This module provides tools for analyzing and visualizing the latent space of trained VAE models using dimensionality
reduction techniques (UMAP, t-SNE).
"""

import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def extract_patient_id_from_filename(filename: str) -> str:
    """Extract patient ID from filename.

    Filename format: ID_HA_YEAR_MONTH_PATIENT.tif
    Example: "1000_HA_2021_02_545.tif" -> "545"

    Args:
        filename: Image filename (e.g., "1000_HA_2021_02_545.tif")

    Returns:
        Patient ID (last part before extension, e.g., "545")
    """
    # Remove extension
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    # Get last part after splitting by underscore
    parts = stem.split("_")
    return parts[-1] if parts else stem


def compute_distance_metrics(points1: np.ndarray, points2: np.ndarray) -> tuple[float, float, float, float] | None:
    """Compute distance metrics between two point clouds.

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
    """Analyzer for VAE latent space using dimensionality reduction.

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

    def __init__(self, vae_model: torch.nn.Module, device: torch.device, transform) -> None:
        self.vae = vae_model
        self.device = device
        self.transform = transform
        self.vae.eval()

    def _apply_pca(self, vectors: np.ndarray, n_components: int = 50) -> tuple[np.ndarray, PCA]:
        """Apply PCA preprocessing to vectors.

        Args:
            vectors: Input vectors [N, D]
            n_components: Number of PCA components

        Returns:
            Tuple of (pca_transformed_vectors, pca_model)
        """
        pca = PCA(n_components=n_components)
        vectors_pca = pca.fit_transform(vectors)
        return vectors_pca, pca

    def encode_images(
        self, image_paths: list[str], max_images: int | None = None, batch_size: int = 8, show_progress: bool = True
    ) -> tuple[np.ndarray, list[str]]:
        """Encode images to latent space with batch processing.

        Args:
            image_paths: List of image file paths
            max_images: Maximum number of images to encode
            batch_size: Batch size for encoding (default: 8)
            show_progress: Whether to show progress bar (default: True)

        Returns:
            Tuple of (latent_vectors [N, D], image_ids)

        Raises:
            ValueError: If image_paths is empty
        """
        if len(image_paths) == 0:
            raise ValueError("image_paths cannot be empty")

        if max_images is not None:
            image_paths = image_paths[:max_images]

        latent_vectors = []
        image_ids = []

        # Setup progress bar
        iterator = range(0, len(image_paths), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Encoding images", unit="batch")
            except ImportError:
                pass  # tqdm not available, continue without progress bar

        with torch.no_grad():
            for i in iterator:
                batch_paths = image_paths[i : i + batch_size]

                # Load and preprocess batch
                batch_imgs = []
                for path in batch_paths:
                    img = self.transform(path)
                    batch_imgs.append(img)

                # Stack and encode batch
                batch_tensor = torch.stack(batch_imgs).to(self.device)
                z = self.vae.encode_stage_2_inputs(batch_tensor)
                z = z.cpu().flatten(start_dim=1)
                latent_vectors.append(z)

                # Extract patient IDs
                for path in batch_paths:
                    filename = os.path.basename(path)
                    image_ids.append(extract_patient_id_from_filename(filename))

        return torch.cat(latent_vectors, dim=0).numpy(), image_ids

    def reduce_dimensionality_umap(
        self,
        latent_vectors: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 40,
        min_dist: float = 0.5,
        random_state: int = 42,
        pca_components: int = 50,
    ) -> tuple[np.ndarray, object]:
        """Reduce dimensionality using PCA + UMAP.

        Args:
            latent_vectors: Input latent vectors [N, D]
            n_components: Number of UMAP components
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance for UMAP
            random_state: Random seed
            pca_components: Number of PCA components before UMAP

        Returns:
            Tuple of (reduced_vectors [N, n_components], umap_model)

        Raises:
            ValueError: If input validation fails
            ImportError: If umap-learn is not installed
        """
        # Input validation
        if latent_vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got {latent_vectors.ndim}D array")

        n_samples = len(latent_vectors)
        if n_samples < pca_components:
            raise ValueError(
                f"Need at least {pca_components} samples for PCA with {pca_components} components, "
                f"got {n_samples} samples. Reduce pca_components or provide more samples."
            )

        if n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors ({n_neighbors}) must be < n_samples ({n_samples}). "
                f"Reduce n_neighbors or provide more samples."
            )

        try:
            import umap
        except ImportError as e:
            raise ImportError("Please install umap-learn: pip install umap-learn") from e

        # PCA preprocessing
        vectors_pca, _ = self._apply_pca(latent_vectors, pca_components)

        # UMAP
        umap_model = umap.UMAP(
            n_components=n_components, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist
        )
        vectors_umap = umap_model.fit_transform(vectors_pca)

        return vectors_umap, umap_model

    def reduce_dimensionality_tsne(
        self,
        latent_vectors: np.ndarray,
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42,
        pca_components: int = 50,
    ) -> np.ndarray:
        """Reduce dimensionality using PCA + t-SNE.

        Args:
            latent_vectors: Input latent vectors [N, D]
            n_components: Number of t-SNE components
            perplexity: Perplexity parameter for t-SNE
            random_state: Random seed
            pca_components: Number of PCA components before t-SNE

        Returns:
            Reduced vectors [N, n_components]

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if latent_vectors.ndim != 2:
            raise ValueError(f"Expected 2D array, got {latent_vectors.ndim}D array")

        n_samples = len(latent_vectors)
        if n_samples < pca_components:
            raise ValueError(
                f"Need at least {pca_components} samples for PCA with {pca_components} components, "
                f"got {n_samples} samples. Reduce pca_components or provide more samples."
            )

        # t-SNE perplexity should be less than n_samples
        if perplexity >= n_samples:
            raise ValueError(
                f"perplexity ({perplexity}) must be < n_samples ({n_samples}). "
                f"Reduce perplexity or provide more samples."
            )

        # Recommended perplexity range
        if perplexity < 5:
            print(f"Warning: perplexity={perplexity} is very low. Consider using 5-50 for better results.")

        # PCA preprocessing
        vectors_pca, _ = self._apply_pca(latent_vectors, pca_components)

        # t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, init="pca", random_state=random_state)
        return tsne.fit_transform(vectors_pca)

    def create_patient_colormap(self, patient_ids: list[str]) -> tuple[dict[str, int], dict[str, str]]:
        """Create color mapping for patients using Plotly colors.

        Args:
            patient_ids: List of patient identifiers

        Returns:
            Tuple of (patient_to_id, patient_to_color) dictionaries where colors are hex strings
        """
        unique_patients = sorted(set(patient_ids))
        patient_to_id = {patient: i for i, patient in enumerate(unique_patients)}

        # Use Plotly color scales - cycle through if more patients than colors
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24
        patient_to_color = {patient: colors[i % len(colors)] for i, patient in enumerate(unique_patients)}

        return patient_to_id, patient_to_color

    def plot_projection_2d(
        self,
        projections: list[tuple[np.ndarray, list[str], str, str]],
        output_path: str,
        title: str = "Latent Space Projection",
        color_by_patient: bool = True,
        show_labels: bool = True,
        image_paths_list: list[list[str]] | None = None,
    ) -> None:
        """Plot 2D projection of latent space using Plotly for interactive visualization.

        Args:
            projections: List of (vectors, ids, marker, name) tuples
            output_path: Path to save figure (supports .html, .png, .jpg, .svg, .pdf)
            title: Plot title
            color_by_patient: If True, color by patient; if False, color by group
            show_labels: If True, show patient ID labels on hover
            image_paths_list: Optional list of image paths for each projection group
        """
        fig = go.Figure()

        # Create colormap if needed
        if color_by_patient:
            all_ids = []
            for _, ids, _, _ in projections:
                all_ids.extend(ids)
            patient_to_id, patient_to_color = self.create_patient_colormap(all_ids)

        # Marker symbols mapping with differentiation for dente/edente
        # 'o' becomes 'circle-open' for edente, 'circle' for dente
        marker_symbols = {
            "o": "circle-open",  # Empty circle for edentulous
            "o_filled": "circle",  # Filled circle for dental
            "^": "triangle-up",
            "s": "square",
            "d": "diamond",
        }

        # Plot each projection
        for proj_idx, (vectors, ids, marker, name) in enumerate(projections):
            x_coords = vectors[:, 0]
            y_coords = vectors[:, 1]

            # Get image paths for this projection group if available
            image_paths = image_paths_list[proj_idx] if image_paths_list and proj_idx < len(image_paths_list) else None

            # Determine marker symbol based on group name
            # Use filled circle for "dente" (dental), open circle for "edente" (edentulous)
            if "dente" in name.lower() and "edente" not in name.lower():
                # This is dental group (with teeth) - use filled circle
                marker_symbol = "circle" if marker == "o" else marker_symbols.get(marker, "circle")
            else:
                # This is edentulous group (without teeth) - use open circle
                marker_symbol = marker_symbols.get(marker, "circle-open")

            if color_by_patient:
                # Color by patient - one trace per patient for better legend
                for exam_id in sorted(set(ids)):
                    mask = [i for i, eid in enumerate(ids) if eid == exam_id]
                    if not mask:
                        continue

                    # Create enriched hover text with image paths
                    hover_text = []
                    for i in mask:
                        text = f"Patient: {ids[i]}<br>Group: {name}<br>Index: {i}"
                        if image_paths and i < len(image_paths):
                            # Add just the filename for readability
                            filename = os.path.basename(image_paths[i])
                            text += f"<br>File: {filename}"
                        hover_text.append(text)

                    fig.add_trace(
                        go.Scatter(
                            x=[x_coords[i] for i in mask],
                            y=[y_coords[i] for i in mask],
                            mode="markers",
                            name=f"Patient {patient_to_id[exam_id]}: {exam_id} ({name})" if show_labels else exam_id,
                            marker={
                                "size": 10,
                                "color": patient_to_color[exam_id],
                                "symbol": marker_symbol,
                                "opacity": 0.7,
                                "line": {"width": 1, "color": "white"},
                            },
                            hovertext=hover_text,
                            hoverinfo="text",
                            showlegend=True,
                            customdata=[image_paths[i] if image_paths and i < len(image_paths) else None for i in mask],
                        )
                    )
            else:
                # Color by group - one trace per group
                hover_text = []
                customdata = []
                for i in range(len(ids)):
                    text = f"Patient: {ids[i]}<br>Group: {name}<br>Index: {i}"
                    if image_paths and i < len(image_paths):
                        filename = os.path.basename(image_paths[i])
                        text += f"<br>File: {filename}"
                    hover_text.append(text)
                    customdata.append(image_paths[i] if image_paths and i < len(image_paths) else None)

                fig.add_trace(
                    go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode="markers",
                        name=name,
                        marker={
                            "size": 10,
                            "symbol": marker_symbol,
                            "opacity": 0.7,
                            "line": {"width": 1, "color": "white"},
                        },
                        hovertext=hover_text,
                        hoverinfo="text",
                        showlegend=True,
                        customdata=customdata,
                    )
                )

        # Update layout
        fig.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 18}},
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=1000,
            height=800,
            template="plotly_white",
            hovermode="closest",
            legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 1.01, "font": {"size": 10}},
        )

        # Save figure
        if output_path.endswith(".html"):
            fig.write_html(output_path)
        else:
            # For static images (PNG, JPG, SVG, PDF) - requires kaleido
            try:
                fig.write_image(output_path, width=1000, height=800, scale=2)
            except Exception as e:
                # Fallback to HTML if static export fails
                html_path = output_path.rsplit(".", 1)[0] + ".html"
                fig.write_html(html_path)
                print(f"Warning: Could not save as {output_path}. Saved as {html_path} instead.")
                print(f"Error: {e}")
                print("To enable PNG/JPG export, install: pip install kaleido")

    def compute_group_statistics(
        self,
        projections: list[tuple[np.ndarray, list[str], str]],
        latent_vectors_list: list[tuple[np.ndarray, list[str], str]],
        output_dir: Path,
    ) -> None:
        """Compute and save statistics for grouped data.

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
                f.write(
                    f"  - [Latent] center_dist: {metrics_lat[0]:.3f}, "
                    f"std_{name1}: {metrics_lat[1]:.3f}, std_{name2}: {metrics_lat[2]:.3f}, "
                    f"mean_cross_dist: {metrics_lat[3]:.3f}\n"
                )
                f.write(
                    f"  - [Projection] center_dist: {metrics_proj[0]:.3f}, "
                    f"std_{name1}: {metrics_proj[1]:.3f}, std_{name2}: {metrics_proj[2]:.3f}, "
                    f"mean_cross_dist: {metrics_proj[3]:.3f}\n\n"
                )

                exam_distances.append((exam, metrics_lat[0]))

        # Sort by distance
        exam_distances.sort(key=lambda x: x[1])
        sorted_file = output_dir / "exams_sorted_by_distance.txt"
        with open(sorted_file, "w") as f:
            f.write("Exams sorted by latent space center distance\n")
            f.write("=" * 60 + "\n\n")
            for exam, dist in exam_distances:
                f.write(f"{exam}: {dist:.3f}\n")

    def save_color_legend(self, exam_to_id: dict[str, int], exam_to_color: dict[str, str], output_path: Path) -> None:
        """Save color legend to file.

        Args:
            exam_to_id: Mapping from exam ID to integer
            exam_to_color: Mapping from exam ID to hex color string
            output_path: Path to save legend
        """
        with open(output_path, "w") as f:
            f.write("Color Legend for Exams\n")
            f.write("=" * 60 + "\n\n")
            for exam in sorted(exam_to_id.keys(), key=lambda x: exam_to_id[x]):
                color_hex = exam_to_color[exam]
                idx = exam_to_id[exam]
                f.write(f"{idx}: {exam} — {color_hex}\n")


def load_image_paths(data_dir: str, max_images: int | None = None, extensions: list[str] | None = None) -> list[str]:
    """Load image paths from directory with flexible extension support.

    Args:
        data_dir: Directory containing images
        max_images: Maximum number of images to load
        extensions: List of file extensions to search for (default: [".tif", ".tiff"])

    Returns:
        List of image paths sorted alphabetically

    Example:
        >>> # Load TIF files only (default)
        >>> paths = load_image_paths("./data")
        >>>
        >>> # Load multiple formats
        >>> paths = load_image_paths("./data", extensions=[".tif", ".png", ".jpg"])
    """
    if extensions is None:
        extensions = [".tif", ".tiff"]

    paths = []
    for ext in extensions:
        # Support both with and without leading dot
        if not ext.startswith("."):
            ext = f".{ext}"
        paths.extend(glob(os.path.join(data_dir, f"*{ext}")))

    # Sort to ensure consistent ordering
    paths = sorted(paths)

    if max_images is not None:
        paths = paths[:max_images]

    return paths
