from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from pti_ldm_vae_v2.analysis.latent_analysis import LatentCache
from pti_ldm_vae_v2.analysis.latent_analysis import LatentSpaceAnalyzer
from pti_ldm_vae_v2.analysis.latent_analysis import compute_group_statistics
from pti_ldm_vae_v2.analysis.latent_analysis import limit_threadpools
from pti_ldm_vae_v2.analysis.latent_analysis import list_image_paths
from pti_ldm_vae_v2.analysis.latent_analysis import save_color_legend
from pti_ldm_vae_v2.vae.config import load_config_and_model
from pti_ldm_vae_v2.vae_regression_common import build_preprocess_transform
from pti_ldm_vae_v2.vae_regression_common import init_device_and_seed
from pti_ldm_vae_v2.vae_regression_common import resolve_run_output_dir

DEFAULT_DPI = 300
DEFAULT_METHOD = "umap"
DEFAULT_NUM_SAMPLES = 1000
DEFAULT_TSNE_PERPLEXITY = 30
DEFAULT_UMAP_MIN_DIST = 0.5
DEFAULT_UMAP_N_NEIGHBORS = 40


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Static VAE latent space analysis")
    parser.add_argument("-c", "--config-file", type=str, required=True, help="Path to the VAE config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the VAE checkpoint")
    parser.add_argument("--folder-edente", type=str, required=True, help="Folder containing edentulous images")
    parser.add_argument("--folder-dente", type=str, default=None, help="Optional folder containing dentulous images")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Maximum number of images per group",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["umap", "tsne"],
        default=DEFAULT_METHOD,
        help="Dimensionality reduction method",
    )
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_UMAP_N_NEIGHBORS, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST, help="UMAP min_dist")
    parser.add_argument("--perplexity", type=int, default=DEFAULT_TSNE_PERPLEXITY, help="t-SNE perplexity")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output DPI for PNG export")
    parser.add_argument("--subtitle", type=str, default=None, help="Optional subtitle for the plot")
    return parser.parse_args()


def _build_title(method: str, has_dente: bool, subtitle: str | None) -> str:
    """Build the plot title string.

    Args:
        method (str): Dimensionality reduction method.
        has_dente (bool): Whether a second group is present.
        subtitle (str | None): Optional subtitle.

    Returns:
        str: Plot title.
    """
    base_title = method.upper()
    if has_dente:
        base_title = f"{base_title} (● dente, ○ edente)"
    if subtitle:
        return f"{base_title}<br><sub>{subtitle}</sub>"
    return base_title


def _create_static_figure(
    *,
    method: str,
    proj_edente: np.ndarray,
    ids_edente: list[str],
    proj_dente: np.ndarray | None,
    ids_dente: list[str] | None,
    patient_to_color: dict[str, str],
    subtitle: str | None,
    scale_factor: float,
) -> go.Figure:
    """Create a Plotly figure for static export.

    Args:
        method (str): Dimensionality reduction method.
        proj_edente (np.ndarray): Projection for edente points.
        ids_edente (list[str]): Patient IDs for edente points.
        proj_dente (np.ndarray | None): Projection for dente points (optional).
        ids_dente (list[str] | None): Patient IDs for dente points.
        patient_to_color (dict[str, str]): Patient -> color mapping.
        subtitle (str | None): Optional subtitle.
        scale_factor (float): Scaling factor applied to sizes for DPI.

    Returns:
        go.Figure: Plotly figure instance.
    """
    marker_size = int(10 * scale_factor)
    line_width = max(1, int(1 * scale_factor))

    fig = go.Figure()

    colors_edente = [patient_to_color.get(pid, "#333333") for pid in ids_edente]
    fig.add_trace(
        go.Scatter(
            x=proj_edente[:, 0],
            y=proj_edente[:, 1],
            mode="markers",
            name="edente",
            marker={
                "size": marker_size,
                "color": colors_edente,
                "symbol": "circle-open",
                "opacity": 0.7,
                "line": {"width": line_width, "color": "white"},
            },
            showlegend=False,
        )
    )

    if proj_dente is not None and ids_dente is not None:
        colors_dente = [patient_to_color.get(pid, "#333333") for pid in ids_dente]
        fig.add_trace(
            go.Scatter(
                x=proj_dente[:, 0],
                y=proj_dente[:, 1],
                mode="markers",
                name="dente",
                marker={
                    "size": marker_size,
                    "color": colors_dente,
                    "symbol": "circle",
                    "opacity": 0.7,
                    "line": {"width": line_width, "color": "white"},
                },
                showlegend=False,
            )
        )

    title_text = _build_title(method, proj_dente is not None, subtitle)
    title_font_size = int(24 * scale_factor)
    axis_font_size = int(18 * scale_factor)
    tick_font_size = int(14 * scale_factor)

    fig.update_layout(
        title={"text": title_text, "x": 0.5, "xanchor": "center", "font": {"size": title_font_size}},
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=1600,
        height=1400,
        template="plotly_white",
        font={"size": tick_font_size},
        xaxis={
            "showgrid": True,
            "gridwidth": scale_factor,
            "gridcolor": "lightgray",
            "title": {"font": {"size": axis_font_size}},
        },
        yaxis={
            "showgrid": True,
            "gridwidth": scale_factor,
            "gridcolor": "lightgray",
            "title": {"font": {"size": axis_font_size}},
        },
    )
    return fig


def _clamp_umap_neighbors(n_neighbors: int, n_samples: int) -> int:
    """Clamp UMAP neighbors to a valid range.

    Args:
        n_neighbors (int): Requested neighbors.
        n_samples (int): Number of samples.

    Returns:
        int: Valid n_neighbors value.
    """
    min_neighbors = 2
    if n_samples <= min_neighbors:
        min_neighbors = max(1, n_samples - 1)
    max_neighbors = max(min_neighbors, min(200, n_samples - 1))
    return max(min_neighbors, min(n_neighbors, max_neighbors))


def _clamp_tsne_perplexity(perplexity: int, n_samples: int) -> int:
    """Clamp t-SNE perplexity to a valid range.

    Args:
        perplexity (int): Requested perplexity.
        n_samples (int): Number of samples.

    Returns:
        int: Valid perplexity value.
    """
    if n_samples <= 1:
        raise ValueError("t-SNE requires at least 2 samples.")
    return max(1, min(perplexity, n_samples - 1))


def main() -> None:
    """Run static latent space analysis and export a PNG (or HTML fallback)."""
    args = parse_args()

    device = init_device_and_seed(seed=None, print_monai_config=False)
    config, vae = load_config_and_model(args.config_file, args.checkpoint, device)
    patch_size = tuple(config.autoencoder_train["patch_size"])
    transforms = build_preprocess_transform(patch_size)
    analyzer = LatentSpaceAnalyzer(vae, device, transforms)

    output_dir = resolve_run_output_dir(
        config.run_dir,
        args.folder_edente,
        args.output_dir,
        "analysis/static",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(config.run_dir) / "analysis" / "latents_cache"
    cache = LatentCache(cache_dir)

    paths_edente = list_image_paths(args.folder_edente, max_images=args.num_samples)
    latent_edente, ids_edente, _ = cache.get_or_encode_batch(
        image_paths=paths_edente,
        analyzer=analyzer,
        checkpoint_path=args.checkpoint,
        patch_size=patch_size,
        group_name="edente",
    )

    latent_dente: np.ndarray | None = None
    ids_dente: list[str] | None = None
    if args.folder_dente:
        paths_dente = list_image_paths(args.folder_dente, max_images=args.num_samples)
        latent_dente, ids_dente, _ = cache.get_or_encode_batch(
            image_paths=paths_dente,
            analyzer=analyzer,
            checkpoint_path=args.checkpoint,
            patch_size=patch_size,
            group_name="dente",
        )

    all_ids = ids_edente + (ids_dente if ids_dente else [])
    patient_to_id, patient_to_color = analyzer.create_patient_colormap(all_ids)
    save_color_legend(patient_to_id, patient_to_color, output_dir / "color_legend.txt")

    proj_dente: np.ndarray | None = None
    if args.method == "umap":
        n_neighbors = _clamp_umap_neighbors(args.n_neighbors, latent_edente.shape[0])
        with limit_threadpools(1):
            proj_edente, umap_model, pca = analyzer.reduce_dimensionality_umap(
                latent_edente,
                n_neighbors=n_neighbors,
                min_dist=float(args.min_dist),
                random_state=None,
                pca_components=50,
            )
            if latent_dente is not None:
                try:
                    proj_dente = umap_model.transform(pca.transform(latent_dente))
                except Exception:
                    proj_dente, _, _ = analyzer.reduce_dimensionality_umap(
                        latent_dente,
                        n_neighbors=n_neighbors,
                        min_dist=float(args.min_dist),
                        random_state=None,
                        pca_components=50,
                    )

        output_filename = "umap_projection.png"
    else:
        total_samples = latent_edente.shape[0] + (latent_dente.shape[0] if latent_dente is not None else 0)
        perplexity = _clamp_tsne_perplexity(args.perplexity, total_samples)
        with limit_threadpools(1):
            combined = (
                np.concatenate([latent_edente, latent_dente], axis=0)
                if latent_dente is not None
                else latent_edente
            )
            proj_all = analyzer.reduce_dimensionality_tsne(
                combined,
                perplexity=perplexity,
                random_state=None,
                pca_components=50,
            )

        split_idx = latent_edente.shape[0]
        proj_edente = proj_all[:split_idx]
        proj_dente = proj_all[split_idx:] if latent_dente is not None else None
        output_filename = "tsne_projection.png"

    scale_factor = args.dpi / 100.0
    fig = _create_static_figure(
        method=args.method,
        proj_edente=proj_edente,
        ids_edente=ids_edente,
        proj_dente=proj_dente,
        ids_dente=ids_dente,
        patient_to_color=patient_to_color,
        subtitle=args.subtitle,
        scale_factor=scale_factor,
    )

    output_path = output_dir / output_filename
    try:
        fig.write_image(
            str(output_path),
            width=int(1600 * scale_factor),
            height=int(1400 * scale_factor),
            scale=1.0,
        )
    except Exception as exc:
        html_path = output_dir / output_filename.replace(".png", ".html")
        fig.write_html(str(html_path))
        print(f"PNG export failed, HTML saved instead: {html_path}")
        print(f"Error: {exc}")

    if proj_dente is not None and latent_dente is not None and ids_dente is not None:
        compute_group_statistics(
            projection_group1=proj_edente,
            projection_group2=proj_dente,
            latent_group1=latent_edente,
            latent_group2=latent_dente,
            ids_group1=ids_edente,
            ids_group2=ids_dente,
            group1_name="edente",
            group2_name="dente",
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
