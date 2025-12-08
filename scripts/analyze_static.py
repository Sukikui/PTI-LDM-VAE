import argparse

import numpy as np
import plotly.graph_objects as go
import umap
from sklearn.decomposition import PCA

from pti_ldm_vae.analysis import LatentSpaceAnalyzer
from pti_ldm_vae.analysis.common import (
    compute_and_save_statistics,
    create_transforms,
    load_and_encode_group_with_cache,
    load_vae_model,
    set_seed,
    setup_device_and_output,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Static Latent Space Analysis (UMAP or t-SNE)")
    parser.add_argument("--vae-weights", type=str, required=True, help="Path to VAE weights file")
    parser.add_argument("--config-file", type=str, required=True, help="Path to model config file")
    parser.add_argument(
        "--folder-edente",
        type=str,
        required=True,
        help="Path to edentulous image group folder",
    )
    parser.add_argument(
        "--folder-dente",
        type=str,
        default=None,
        help="Path to dentulous image group folder (optional)",
    )
    parser.add_argument("--output-dir", type=str, default="projections", help="Output directory for projections")
    parser.add_argument("--max-images", type=int, default=1000, help="Maximum number of images per group")
    parser.add_argument("--patch-size", type=int, nargs=2, default=[256, 256], help="Image patch size (H W)")
    parser.add_argument(
        "--color-by-patient",
        action="store_true",
        help="Color points by patient ID instead of group",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["umap", "tsne"],
        default="umap",
        help="Dimensionality reduction method (default: umap)",
    )
    parser.add_argument("--n-neighbors", type=int, default=40, help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.5, help="UMAP min_dist parameter")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--subtitle", type=str, default=None, help="Optional subtitle for the plot")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output PNG (default: 300)")
    return parser.parse_args()


def create_plotly_figure(
    projections: list,
    patient_to_id: dict,
    patient_to_color: dict,
    color_by_patient: bool,
    title: str,
    subtitle: str | None,
    scale_factor: float = 3.0,
) -> go.Figure:
    """Create Plotly figure for static high-resolution visualization.

    Args:
        projections: List of (vectors, ids, marker, name) tuples
        patient_to_id: Mapping from patient ID to integer
        patient_to_color: Mapping from patient ID to color
        color_by_patient: Whether to color by patient
        title: Plot title
        subtitle: Optional subtitle
        scale_factor: Scale factor for high-res export (default: 3.0 for 300 DPI)

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Marker symbols mapping
    marker_symbols = {
        "o": "circle-open",  # Empty circle for edentulous
        "o_filled": "circle",  # Filled circle for dental
    }

    # Scale marker size and line width for high-res
    marker_size = int(10 * scale_factor)
    line_width = int(1 * scale_factor)

    for proj_idx, (vectors, ids, marker, name) in enumerate(projections):
        x_coords = vectors[:, 0]
        y_coords = vectors[:, 1]

        # Determine marker symbol
        if "dente" in name.lower() and "edente" not in name.lower():
            marker_symbol = "circle"  # Filled for dente
        else:
            marker_symbol = "circle-open"  # Empty for edente

        if color_by_patient:
            # Plot each patient separately with different colors
            for exam_id in sorted(set(ids)):
                mask = [i for i, eid in enumerate(ids) if eid == exam_id]
                if not mask:
                    continue

                x_patient = [x_coords[i] for i in mask]
                y_patient = [y_coords[i] for i in mask]

                fig.add_trace(
                    go.Scatter(
                        x=x_patient,
                        y=y_patient,
                        mode="markers",
                        name=f"{name}: {exam_id}",
                        marker={
                            "size": marker_size,
                            "color": patient_to_color[exam_id],
                            "symbol": marker_symbol,
                            "opacity": 0.7,
                            "line": {"width": line_width, "color": "white"},
                        },
                        showlegend=False,  # No legend in static version
                    )
                )
        else:
            # Plot whole group with single color
            color = "#1f77b4" if proj_idx == 0 else "#ff7f0e"

            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers",
                    name=name,
                    marker={
                        "size": marker_size,
                        "color": color,
                        "symbol": marker_symbol,
                        "opacity": 0.7,
                        "line": {"width": line_width, "color": "white"},
                    },
                    showlegend=False,  # No legend in static version
                )
            )

    # Set title
    title_text = title
    if subtitle:
        title_text = f"{title}<br><sub>{subtitle}</sub>"

    # Scale font sizes for high-res
    title_font_size = int(24 * scale_factor)
    axis_font_size = int(18 * scale_factor)
    tick_font_size = int(14 * scale_factor)

    # Update layout for high-quality static export
    fig.update_layout(
        title={
            "text": title_text,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": title_font_size},
        },
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        width=1600,  # Base resolution
        height=1400,  # Base resolution
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


def main() -> None:
    """Run static latent space analysis and generate high-res PNG visualizations.

    This script generates static PNG files for latent space visualization using either UMAP or t-SNE. Unlike the
    interactive version, this script saves output files and exits (no server).
    """
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print(f"Static Latent Space Analysis - {args.method.upper()}")
    print("=" * 60)

    # Setup
    device, output_dir = setup_device_and_output(args.output_dir)
    vae = load_vae_model(args.config_file, args.vae_weights, device)
    transforms = create_transforms(tuple(args.patch_size))
    analyzer = LatentSpaceAnalyzer(vae, device, transforms)

    # Load and encode groups (with caching)
    print("\n" + "=" * 60)
    latent_edente, ids_edente, paths_edente = load_and_encode_group_with_cache(
        analyzer=analyzer,
        folder_path=args.folder_edente,
        vae_weights=args.vae_weights,
        max_images=args.max_images,
        patch_size=tuple(args.patch_size),
        group_name="edente",
    )

    latent_dente, ids_dente, paths_dente = None, None, None
    if args.folder_dente:
        latent_dente, ids_dente, paths_dente = load_and_encode_group_with_cache(
            analyzer=analyzer,
            folder_path=args.folder_dente,
            vae_weights=args.vae_weights,
            max_images=args.max_images,
            patch_size=tuple(args.patch_size),
            group_name="dente",
        )

    # Dimensionality reduction
    print("\n" + "=" * 60)
    print(f"Computing {args.method.upper()} projection...")
    print("=" * 60)

    if args.method == "umap":
        # Fit PCA on the first group
        pca = PCA(n_components=50, random_state=args.seed)
        latent_edente_pca = pca.fit_transform(latent_edente)
        print(f"✅ PCA fitted on edente: {latent_edente_pca.shape}")

        # Fit UMAP on the PCA-transformed first group
        umap_model = umap.UMAP(
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            random_state=args.seed,
            n_components=2,
        ).fit(latent_edente_pca)
        proj_edente = umap_model.embedding_
        print(f"✅ UMAP fitted on edente: {proj_edente.shape}")

        projections = [(proj_edente, ids_edente, "o", "edente")]

        # Transform dente using fitted UMAP model if provided
        if args.folder_dente:
            latent_dente_pca = pca.transform(latent_dente)
            proj_dente = umap_model.transform(latent_dente_pca)
            print(f"✅ UMAP transformed dente: {proj_dente.shape}")
            projections.append((proj_dente, ids_dente, "o_filled", "dente"))

        output_filename = "umap_projection.png"
        title = "UMAP"

    else:  # tsne
        print("(This may take a few minutes...)")

        # Combine all data for joint t-SNE
        combined_latent = np.concatenate([latent_edente, latent_dente]) if args.folder_dente else latent_edente

        tsne_combined = analyzer.reduce_dimensionality_tsne(
            combined_latent, perplexity=args.perplexity, random_state=args.seed
        )
        print(f"✅ t-SNE computed: {tsne_combined.shape}")

        # Split back into groups
        split_idx = len(latent_edente)
        proj_edente = tsne_combined[:split_idx]
        projections = [(proj_edente, ids_edente, "o", "edente")]

        if args.folder_dente:
            proj_dente = tsne_combined[split_idx:]
            projections.append((proj_dente, ids_dente, "o_filled", "dente"))

        output_filename = "tsne_projection.png"
        title = "t-SNE"

    # Create title with legend symbols
    if args.folder_dente:
        title = f"{title} (● dente, ○ edente)"
    else:
        title = title

    # Create colormap if needed
    patient_to_id, patient_to_color = {}, {}
    if args.color_by_patient:
        all_ids = ids_edente + (ids_dente if ids_dente else [])
        patient_to_id, patient_to_color = analyzer.create_patient_colormap(all_ids)

        # Save color legend
        legend_path = output_dir / "color_legend.txt"
        analyzer.save_color_legend(patient_to_id, patient_to_color, legend_path)
        print(f"✅ Color legend saved: {legend_path}")

    # Create and save visualization
    print("\n" + "=" * 60)
    print("Generating visualization...")
    print("=" * 60)

    # Calculate scale factor based on DPI
    # Plotly default is ~100 DPI, so scale = dpi/100
    scale_factor = args.dpi / 100.0

    fig = create_plotly_figure(
        projections,
        patient_to_id,
        patient_to_color,
        args.color_by_patient,
        title,
        args.subtitle,
        scale_factor=scale_factor,
    )

    output_path = output_dir / output_filename

    print(f"Exporting PNG at {int(1600 * scale_factor)}x{int(1400 * scale_factor)} pixels...")

    try:
        fig.write_image(
            str(output_path),
            width=int(1600 * scale_factor),
            height=int(1400 * scale_factor),
            scale=1.0,  # Don't double-scale
        )
        print(f"✅ Visualization saved: {output_path}")
    except Exception as e:
        # Fallback to HTML if kaleido is not installed
        html_path = output_dir / output_filename.replace(".png", ".html")
        fig.write_html(str(html_path))
        print(f"⚠️  Could not export PNG (kaleido required). Saved HTML instead: {html_path}")
        print("   Install kaleido with: pip install kaleido")
        print(f"   Error: {e}")

    # Compute statistics if two groups
    if args.folder_dente:
        compute_and_save_statistics(
            analyzer,
            proj_edente,
            proj_dente,
            latent_edente,
            latent_dente,
            ids_edente,
            ids_dente,
            "edente",
            "dente",
            output_dir,
        )

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)
    print(f"\nGenerated files in {output_dir}:")
    print(f"  - {output_filename}")
    if args.color_by_patient:
        print("  - color_legend.txt")
    if args.folder_dente:
        print("  - distance_metrics.txt")
        print("  - exams_sorted_by_distance.txt")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
