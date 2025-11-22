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
    load_and_encode_group_with_cache,
    load_vae_model,
    set_seed,
    setup_device_and_output,
)
from dash import dcc, html
from dash.dependencies import Input, Output, State
from PIL import Image
from sklearn.decomposition import PCA

from pti_ldm_vae.analysis import LatentSpaceAnalyzer, latent_distance, latent_distance_cross


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
        help="Path to dentulous image group folder (optional)",
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
    parser.add_argument("--subtitle", type=str, default=None, help="Optional subtitle for the plot")
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
                        name=f"{name}: {exam_id}",
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
        template="plotly_white",
        hovermode="closest",
        clickmode="event",
        margin={"l": 50, "r": 20, "t": 80, "b": 50},
        autosize=True,
    )

    return fig


def main() -> None:
    """Run interactive Dash application for latent space visualization.

    This function creates an interactive web application that displays VAE latent space projections with click-to-view
    image functionality. The app shows UMAP or t-SNE projections on the left and displays clicked images on the right.
    Interactive sliders allow real-time adjustment of dimensionality reduction parameters.
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
        image_paths_list = [paths_edente]

        if args.folder_dente:
            # Transform the second group using the same PCA and UMAP models
            latent_dente_pca = pca.transform(latent_dente)
            proj_dente = umap_model.transform(latent_dente_pca)
            print(f"✅ UMAP transformed dente: {proj_dente.shape}")
            projections.append((proj_dente, ids_dente, "o", "dente"))
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
            projections.append((proj_dente, ids_dente, "o", "dente"))
            image_paths_list.append(paths_dente)

    # Create colormap if needed
    patient_to_id, patient_to_color = {}, {}
    if args.color_by_patient:
        all_ids = ids_edente + (ids_dente if ids_dente else [])
        patient_to_id, patient_to_color = analyzer.create_patient_colormap(all_ids)

    # Create title with legend
    if args.folder_dente:
        title = f"{args.method.upper()} (● dente, ○ edente)"
    else:
        title = args.method.upper()

    # Add subtitle if provided
    if args.subtitle:
        title = f"{title}<br><sub>{args.subtitle}</sub>"

    # Create Dash app
    print("\n" + "=" * 60)
    print("Creating Dash app...")
    print("=" * 60)

    app = dash.Dash(__name__)

    # Add global CSS to remove browser default margins/padding
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body {
                    margin: 0 !important;
                    padding: 0 !important;
                    overflow: hidden !important;
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """

    # Create parameter controls based on method
    if args.method == "umap":
        param_controls = html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "n_neighbors",
                            style={
                                "fontWeight": "bold",
                                "marginBottom": "5px",
                                "marginTop": "10px",
                                "display": "block",
                                "paddingLeft": "20px",
                            },
                        ),
                        dcc.Slider(
                            id="n-neighbors-slider",
                            min=5,
                            max=200,
                            step=5,
                            value=args.n_neighbors,
                            marks={5: "5", 50: "50", 100: "100", 150: "150", 200: "200"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(
                    [
                        html.Label(
                            "min_dist",
                            style={
                                "fontWeight": "bold",
                                "marginBottom": "5px",
                                "marginTop": "10px",
                                "display": "block",
                                "paddingLeft": "20px",
                            },
                        ),
                        dcc.Slider(
                            id="min-dist-slider",
                            min=0.0,
                            max=0.99,
                            step=0.05,
                            value=args.min_dist,
                            marks={0.0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 0.99: "0.99"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
            ],
            style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#f9f9f9", "borderRadius": "5px"},
        )
    else:  # tsne
        param_controls = html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "perplexity",
                            style={
                                "fontWeight": "bold",
                                "marginBottom": "5px",
                                "marginTop": "10px",
                                "display": "block",
                                "paddingLeft": "20px",
                            },
                        ),
                        dcc.Slider(
                            id="perplexity-slider",
                            min=5,
                            max=min(100, len(latent_edente) // 3),
                            step=5,
                            value=args.perplexity,
                            marks={
                                5: "5",
                                15: "15",
                                30: "30",
                                50: "50",
                                min(100, len(latent_edente) // 3): str(min(100, len(latent_edente) // 3)),
                            },
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ],
                    style={"marginBottom": "10px"},
                ),
            ],
            style={"marginBottom": "20px", "padding": "10px", "backgroundColor": "#f9f9f9", "borderRadius": "5px"},
        )

    initial_figure = create_figure(
        projections,
        image_paths_list,
        patient_to_id,
        patient_to_color,
        args.color_by_patient,
        title,
    )

    app.layout = html.Div(
        [
            # Hidden divs to store data
            dcc.Store(
                id="stored-projections",
                data={
                    "projections": None,
                    "method": args.method,
                    "latent_edente": latent_edente.tolist(),
                    "latent_dente": latent_dente.tolist() if latent_dente is not None else None,
                    "ids_edente": ids_edente,
                    "ids_dente": ids_dente,
                    "paths_edente": paths_edente,
                    "paths_dente": paths_dente,
                    "pca_components": None,
                    "seed": args.seed,
                },
            ),
            dcc.Store(id="base-figure", data=initial_figure.to_dict()),
            dcc.Store(id="selection-store", data=None),
            html.Div(
                [
                    # Left: Plot
                    html.Div(
                        [
                            dcc.Graph(
                                id="latent-plot",
                                figure=initial_figure,
                                style={"height": "100%", "width": "100%"},
                            )
                        ],
                        style={"flex": "4", "paddingRight": "10px", "display": "flex"},
                    ),
                    # Right: Controls and Image viewer
                    html.Div(
                        [
                            html.H3("Parameters", style={"textAlign": "center", "marginBottom": "15px"}),
                            param_controls,
                            html.H3("Selected Image", style={"textAlign": "center", "marginTop": "60px"}),
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
                            html.Div(
                                id="distance-info",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "10px",
                                    "fontSize": "13px",
                                    "color": "#444",
                                    "padding": "5px",
                                    "border": "1px solid #eee",
                                    "borderRadius": "4px",
                                },
                            ),
                        ],
                        style={
                            "flex": "1",  # Takes 1/5 of the space
                            "padding": "20px",
                            "borderLeft": "1px solid #ddd",
                            "overflowY": "auto",
                        },
                    ),
                ],
                style={"display": "flex", "flexDirection": "row", "height": "100vh"},
            ),
        ],
        style={
            "fontFamily": "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
            "margin": "0",
            "padding": "0",
            "overflow": "hidden",
        },
    )

    # Figure update with optional selection overlay
    if args.method == "umap":

        @app.callback(
            Output("base-figure", "data"),
            [Input("n-neighbors-slider", "value"), Input("min-dist-slider", "value")],
        )
        def update_base_figure_umap(n_neighbors_val, min_dist_val):
            print("\n" + "=" * 60)
            print("Recalculating UMAP projection...")
            print(f"n_neighbors: {n_neighbors_val}, min_dist: {min_dist_val}")
            print("=" * 60)

            pca = PCA(n_components=50, random_state=args.seed)
            latent_edente_pca = pca.fit_transform(latent_edente)

            umap_model = umap.UMAP(
                n_neighbors=n_neighbors_val,
                min_dist=min_dist_val,
                random_state=args.seed,
                n_components=2,
            ).fit(latent_edente_pca)

            new_proj_edente = umap_model.embedding_
            new_projections = [(new_proj_edente, ids_edente, "o", "edente")]
            new_image_paths_list = [paths_edente]

            if args.folder_dente:
                latent_dente_pca = pca.transform(latent_dente)
                new_proj_dente = umap_model.transform(latent_dente_pca)
                new_projections.append((new_proj_dente, ids_dente, "o", "dente"))
                new_image_paths_list.append(paths_dente)

            fig = create_figure(
                new_projections,
                new_image_paths_list,
                patient_to_id,
                patient_to_color,
                args.color_by_patient,
                title,
            )
            return fig.to_dict()
    else:

        @app.callback(
            Output("base-figure", "data"),
            [Input("perplexity-slider", "value")],
        )
        def update_base_figure_tsne(perplexity_val):
            print("\n" + "=" * 60)
            print("Recalculating t-SNE projection...")
            print(f"perplexity: {perplexity_val}")
            print("=" * 60)

            combined_latent = np.concatenate([latent_edente, latent_dente]) if args.folder_dente else latent_edente

            tsne_combined = analyzer.reduce_dimensionality_tsne(
                combined_latent, perplexity=perplexity_val, random_state=args.seed
            )

            split_idx = len(latent_edente)
            new_proj_edente = tsne_combined[:split_idx]
            new_projections = [(new_proj_edente, ids_edente, "o", "edente")]
            new_image_paths_list = [paths_edente]

            if args.folder_dente:
                new_proj_dente = tsne_combined[split_idx:]
                new_projections.append((new_proj_dente, ids_dente, "o", "dente"))
                new_image_paths_list.append(paths_dente)

            fig = create_figure(
                new_projections,
                new_image_paths_list,
                patient_to_id,
                patient_to_color,
                args.color_by_patient,
                title,
            )
            return fig.to_dict()

    @app.callback(
        Output("latent-plot", "figure"),
        [Input("base-figure", "data"), Input("selection-store", "data")],
    )
    def overlay_selection(base_fig_data, selection_data):
        if base_fig_data is None:
            return go.Figure()
        fig = go.Figure(base_fig_data)
        if selection_data and selection_data.get("points") and len(selection_data["points"]) == 2:
            p1, p2 = selection_data["points"]
            fig.add_trace(
                go.Scatter(
                    x=[p1["x"], p2["x"]],
                    y=[p1["y"], p2["y"]],
                    mode="lines",
                    line={"color": "#888", "width": 1},
                    name="selected-pair",
                    showlegend=False,
                )
            )
        return fig

    @app.callback(
        [
            Output("image-container", "children"),
            Output("image-info", "children"),
            Output("selection-store", "data"),
            Output("distance-info", "children"),
        ],
        [Input("latent-plot", "clickData")],
        [State("selection-store", "data")],
    )
    def display_click_image(clickData, selection_data):
        if clickData is None:
            return html.Div("Click on a point to view the image"), "", selection_data, ""

        try:
            point = clickData["points"][0]
            customdata = point.get("customdata", {})

            if not customdata or not customdata.get("path"):
                return html.Div("No image path available"), "", selection_data, ""

            image_path = customdata["path"]
            patient = customdata.get("patient", "Unknown")
            group = customdata.get("group", "Unknown")
            index = int(customdata.get("index", -1))
            x_coord = float(point.get("x"))
            y_coord = float(point.get("y"))

            img_src = load_image_as_base64(image_path)
            if not img_src:
                return html.Div(f"Error loading image: {image_path}"), "", selection_data, ""

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

            image_div = html.Img(
                src=img_src,
                style={
                    "maxWidth": "100%",
                    "maxHeight": "500px",
                    "border": "2px solid #ddd",
                    "borderRadius": "5px",
                },
            )

            # Update selection
            new_point = {
                "group": group,
                "index": index,
                "x": x_coord,
                "y": y_coord,
                "patient": patient,
                "path": image_path,
            }
            points = []
            if selection_data and selection_data.get("points"):
                points = selection_data["points"]
            if len(points) == 0:
                points = [new_point]
                return image_div, info, {"points": points}, ""
            if len(points) == 1:
                points.append(new_point)
            else:
                points = [new_point]
                return image_div, info, {"points": points}, ""

            # Compute distance when two points selected
            p1, p2 = points
            distance_val = None
            projection_distance = None
            try:
                if p1["group"] == "edente":
                    lat1 = latent_edente[p1["index"]]
                else:
                    lat1 = latent_dente[p1["index"]] if latent_dente is not None else None
                if p2["group"] == "edente":
                    lat2 = latent_edente[p2["index"]]
                else:
                    lat2 = latent_dente[p2["index"]] if latent_dente is not None else None
                if lat1 is None or lat2 is None:
                    distance_val = None
                elif p1["group"] == p2["group"]:
                    distance_val = latent_distance(lat1, lat2)
                else:
                    distance_val = latent_distance_cross(
                        latent_edente if p1["group"] == "edente" else latent_dente,
                        p1["index"],
                        latent_dente if p1["group"] == "edente" else latent_edente,
                        p2["index"],
                    )
            except Exception as exc:  # pylint: disable=broad-except
                distance_val = None
                print(f"[WARN] Failed to compute latent distance: {exc}")

            try:
                projection_distance = float(np.linalg.norm(np.array([p1["x"], p1["y"]]) - np.array([p2["x"], p2["y"]])))
            except Exception as exc:  # pylint: disable=broad-except
                projection_distance = None
                print(f"[WARN] Failed to compute projection distance: {exc}")

            if distance_val is None:
                distance_block = html.Div("Distance non disponible")
            else:
                group_pair = (
                    f"{p1['group']} ↔ {p2['group']}" if p1["group"] != p2["group"] else f"{p1['group']} (intra)"
                )
                distance_block = html.Div(
                    [
                        html.Div(
                            [
                                html.P(
                                    f"Distance latente: {distance_val:.4f}",
                                    style={"margin": "4px 0", "fontWeight": "bold"},
                                ),
                                html.P(
                                    f"Distance projection (2D): {projection_distance:.4f}"
                                    if projection_distance is not None
                                    else "Distance projection (2D): n/a",
                                    style={"margin": "3px 0"},
                                ),
                                html.P(group_pair, style={"margin": "3px 0", "color": "#666"}),
                            ],
                            style={"marginBottom": "6px"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.P("P1", style={"margin": "2px 0", "fontWeight": "bold"}),
                                        html.P(f"Groupe: {p1['group']}", style={"margin": "2px 0"}),
                                        html.P(f"Patient: {p1['patient']}", style={"margin": "2px 0"}),
                                        html.P(f"Index: {p1['index']}", style={"margin": "2px 0"}),
                                    ],
                                    style={
                                        "flex": "1",
                                        "padding": "6px",
                                        "backgroundColor": "#f9f9f9",
                                        "borderRadius": "4px",
                                        "marginRight": "6px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.P("P2", style={"margin": "2px 0", "fontWeight": "bold"}),
                                        html.P(f"Groupe: {p2['group']}", style={"margin": "2px 0"}),
                                        html.P(f"Patient: {p2['patient']}", style={"margin": "2px 0"}),
                                        html.P(f"Index: {p2['index']}", style={"margin": "2px 0"}),
                                    ],
                                    style={
                                        "flex": "1",
                                        "padding": "6px",
                                        "backgroundColor": "#f9f9f9",
                                        "borderRadius": "4px",
                                    },
                                ),
                            ],
                            style={"display": "flex", "flexDirection": "row"},
                        ),
                    ]
                )

            return image_div, info, {"points": points}, distance_block

        except Exception as e:
            return html.Div(f"Error: {e!s}"), "", selection_data, ""

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
