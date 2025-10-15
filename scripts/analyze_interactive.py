import argparse
import base64
from io import BytesIO
from pathlib import Path

import dash
import numpy as np
import plotly.graph_objects as go
import umap
from analysis_common import (
    create_transforms,
    load_and_encode_group,
    load_vae_model,
    set_seed,
    setup_device_and_output,
)
from dash import dcc, html
from dash.dependencies import Input, Output
from PIL import Image
from sklearn.decomposition import PCA

from pti_ldm_vae.analysis import LatentSpaceAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive Latent Space Analysis with Dash")
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
        help="Path to dental image group folder (optional)",
    )
    parser.add_argument("--max-images", type=int, default=3000, help="Maximum number of images per group")
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
        help="Dimensionality reduction method",
    )
    parser.add_argument("--n-neighbors", type=int, default=40, help="UMAP n_neighbors parameter")
    parser.add_argument("--min-dist", type=float, default=0.5, help="UMAP min_dist parameter")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash server (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode")
    return parser.parse_args()


def load_image_as_base64(image_path: str, max_size: int = 512) -> str:
    """Load image and convert to base64 for display in Dash.

    Args:
        image_path: Path to image file
        max_size: Maximum width/height for display

    Returns:
        Base64 encoded image string
    """
    try:
        img = Image.open(image_path)

        # Convert to grayscale if needed
        if img.mode != "L":
            img = img.convert("L")

        # Resize for display if too large
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Convert to PNG in memory
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode to base64
        img_base64 = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{img_base64}"

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return ""


def create_figure(
    projections: list,
    image_paths_list: list | None,
    patient_to_id: dict,
    patient_to_color: dict,
    color_by_patient: bool,
    title: str,
) -> go.Figure:
    """Create Plotly figure for Dash.

    Args:
        projections: List of (vectors, ids, marker, name) tuples
        image_paths_list: List of image paths for each group
        patient_to_id: Mapping from patient ID to integer
        patient_to_color: Mapping from patient ID to color
        color_by_patient: Whether to color by patient
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Marker symbols
    marker_symbols = {
        "o": "circle-open",
        "o_filled": "circle",
        "^": "triangle-up",
        "s": "square",
        "d": "diamond",
    }

    for proj_idx, (vectors, ids, marker, name) in enumerate(projections):
        x_coords = vectors[:, 0]
        y_coords = vectors[:, 1]
        image_paths = image_paths_list[proj_idx] if image_paths_list and proj_idx < len(image_paths_list) else None

        # Determine marker symbol
        if "dente" in name.lower() and "edente" not in name.lower():
            marker_symbol = "circle" if marker == "o" else marker_symbols.get(marker, "circle")
        else:
            marker_symbol = marker_symbols.get(marker, "circle-open")

        if color_by_patient:
            for exam_id in sorted(set(ids)):
                mask = [i for i, eid in enumerate(ids) if eid == exam_id]
                if not mask:
                    continue

                hover_text = []
                for i in mask:
                    text = f"Patient: {ids[i]}<br>Group: {name}<br>Index: {i}"
                    if image_paths and i < len(image_paths):
                        filename = Path(image_paths[i]).name
                        text += f"<br>File: {filename}"
                    hover_text.append(text)

                fig.add_trace(
                    go.Scatter(
                        x=[x_coords[i] for i in mask],
                        y=[y_coords[i] for i in mask],
                        mode="markers",
                        name=f"Patient {patient_to_id[exam_id]}: {exam_id} ({name})",
                        marker={
                            "size": 10,
                            "color": patient_to_color[exam_id],
                            "symbol": marker_symbol,
                            "opacity": 0.7,
                            "line": {"width": 1, "color": "white"},
                        },
                        hovertext=hover_text,
                        hoverinfo="text",
                        customdata=[
                            {
                                "path": image_paths[i] if image_paths and i < len(image_paths) else None,
                                "patient": ids[i],
                                "group": name,
                                "index": i,
                            }
                            for i in mask
                        ],
                    )
                )
        else:
            hover_text = []
            customdata = []
            for i in range(len(ids)):
                text = f"Patient: {ids[i]}<br>Group: {name}<br>Index: {i}"
                if image_paths and i < len(image_paths):
                    filename = Path(image_paths[i]).name
                    text += f"<br>File: {filename}"
                hover_text.append(text)
                customdata.append(
                    {
                        "path": image_paths[i] if image_paths and i < len(image_paths) else None,
                        "patient": ids[i],
                        "group": name,
                        "index": i,
                    }
                )

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
                    customdata=customdata,
                )
            )

    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 18}},
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=700,
        template="plotly_white",
        hovermode="closest",
        clickmode="event+select",
    )

    return fig


def main() -> None:
    """Run interactive Dash application for latent space visualization.

    This function creates an interactive web application that displays VAE latent space projections with click-to-view
    image functionality. The app shows UMAP or t-SNE projections on the left and displays clicked images on the right.
    """
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("Interactive Latent Space Analysis")
    print("=" * 60)

    # Setup (output_dir not used but keeps API consistent)
    device, _ = setup_device_and_output("/tmp")
    vae = load_vae_model(args.config_file, args.vae_weights, device)
    transforms = create_transforms(tuple(args.patch_size))
    analyzer = LatentSpaceAnalyzer(vae, device, transforms)

    # Load and encode groups
    print("\n" + "=" * 60)
    latent_edente, ids_edente, paths_edente = load_and_encode_group(
        analyzer, args.folder_edente, args.max_images, "edente"
    )

    latent_dente, ids_dente, paths_dente = None, None, None
    if args.folder_dente:
        latent_dente, ids_dente, paths_dente = load_and_encode_group(
            analyzer, args.folder_dente, args.max_images, "dente"
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
        image_paths_list = [paths_edente]

        if args.folder_dente:
            # Transform the second group using the same PCA and UMAP models
            latent_dente_pca = pca.transform(latent_dente)
            proj_dente = umap_model.transform(latent_dente_pca)
            print(f"✅ UMAP transformed dente: {proj_dente.shape}")
            projections.append((proj_dente, ids_dente, "^", "dente"))
            image_paths_list.append(paths_dente)

    else:  # tsne
        combined_latent = np.concatenate([latent_edente, latent_dente]) if args.folder_dente else latent_edente
        print("(This may take a few minutes...)")
        tsne_combined = analyzer.reduce_dimensionality_tsne(
            combined_latent, perplexity=args.perplexity, random_state=args.seed
        )
        print(f"✅ t-SNE computed: {tsne_combined.shape}")

        split_idx = len(latent_edente)
        proj_edente = tsne_combined[:split_idx]
        projections = [(proj_edente, ids_edente, "o", "edente")]
        image_paths_list = [paths_edente]

        if args.folder_dente:
            proj_dente = tsne_combined[split_idx:]
            projections.append((proj_dente, ids_dente, "^", "dente"))
            image_paths_list.append(paths_dente)

    # Create colormap if needed
    patient_to_id, patient_to_color = {}, {}
    if args.color_by_patient:
        all_ids = ids_edente + (ids_dente if ids_dente else [])
        patient_to_id, patient_to_color = analyzer.create_patient_colormap(all_ids)

    # Create title
    title = args.method.upper()

    # Create Dash app
    print("\n" + "=" * 60)
    print("Creating Dash app...")
    print("=" * 60)

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            html.Div(
                [
                    # Left: Plot
                    html.Div(
                        [
                            dcc.Graph(
                                id="latent-plot",
                                figure=create_figure(
                                    projections,
                                    image_paths_list,
                                    patient_to_id,
                                    patient_to_color,
                                    args.color_by_patient,
                                    title,
                                ),
                                style={"height": "100vh"},  # Full height
                            )
                        ],
                        style={"flex": "3", "paddingRight": "10px"},  # Takes 3/5 of the space
                    ),
                    # Right: Image viewer
                    html.Div(
                        [
                            html.H3("Selected Image", style={"textAlign": "center"}),
                            html.Div(
                                id="image-info",
                                style={
                                    "textAlign": "center",
                                    "marginBottom": "10px",
                                    "fontSize": "14px",
                                },
                            ),
                            html.Div(
                                id="image-container",
                                style={"textAlign": "center", "padding": "20px"},
                            ),
                        ],
                        style={
                            "flex": "2",  # Takes 2/5 of the space
                            "padding": "20px",
                            "borderLeft": "1px solid #ddd",
                            "height": "100vh",
                            "overflowY": "auto",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "row"},
            ),
        ],
        style={
            "fontFamily": "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
            "padding": "10px",
        },
    )

    @app.callback(
        [Output("image-container", "children"), Output("image-info", "children")],
        [Input("latent-plot", "clickData")],
    )
    def display_click_image(clickData):
        if clickData is None:
            return html.Div("Click on a point to view the image"), ""

        try:
            # Get customdata from clicked point
            point = clickData["points"][0]
            customdata = point.get("customdata", {})

            if not customdata or not customdata.get("path"):
                return html.Div("No image path available"), ""

            image_path = customdata["path"]
            patient = customdata.get("patient", "Unknown")
            group = customdata.get("group", "Unknown")
            index = customdata.get("index", "?")

            # Load and encode image
            img_src = load_image_as_base64(image_path)

            if not img_src:
                return html.Div(f"Error loading image: {image_path}"), ""

            # Image info
            info = html.Div(
                [
                    html.P(f"Patient: {patient}", style={"margin": "5px"}),
                    html.P(f"Group: {group}", style={"margin": "5px"}),
                    html.P(f"Index: {index}", style={"margin": "5px"}),
                    html.P(
                        f"File: {Path(image_path).name}",
                        style={"margin": "5px", "fontSize": "11px", "color": "#666"},
                    ),
                ]
            )

            # Image display
            image_div = html.Img(
                src=img_src,
                style={
                    "maxWidth": "100%",
                    "maxHeight": "500px",
                    "border": "2px solid #ddd",
                    "borderRadius": "5px",
                },
            )

            return image_div, info

        except Exception as e:
            return html.Div(f"Error: {e!s}"), ""

    # Run server
    print("\n" + "=" * 60)
    print(f"🚀 Starting Dash server on http://localhost:{args.port}")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server")
    print("\nOpen your browser and navigate to the URL above to view the interactive app.")
    print("=" * 60 + "\n")

    app.run(debug=args.debug, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()
