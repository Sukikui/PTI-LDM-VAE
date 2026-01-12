from __future__ import annotations

import argparse
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import dash
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from PIL import Image

from pti_ldm_vae_v2.vae.config import load_config_and_model
from pti_ldm_vae_v2.analysis.latent_analysis import LatentCache
from pti_ldm_vae_v2.analysis.latent_analysis import LatentSpaceAnalyzer
from pti_ldm_vae_v2.analysis.latent_analysis import limit_threadpools
from pti_ldm_vae_v2.analysis.latent_analysis import latent_distance
from pti_ldm_vae_v2.analysis.latent_analysis import latent_distance_cross
from pti_ldm_vae_v2.analysis.latent_analysis import list_image_paths
from pti_ldm_vae_v2.common import build_preprocess_transform
from pti_ldm_vae_v2.common import init_device_and_seed

DEFAULT_BATCH_SIZE = 8
DEFAULT_METHOD = "tsne"
DEFAULT_NUM_SAMPLES = 3000
DEFAULT_PORT = 8050
DEFAULT_TSNE_PERPLEXITY = 30
DEFAULT_UMAP_MIN_DIST = 0.5
DEFAULT_UMAP_N_NEIGHBORS = 40


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed args.
    """
    parser = argparse.ArgumentParser(description="Interactive VAE latent space analysis (Dash)")
    parser.add_argument("-c", "--config-file", type=str, required=True, help="Path to the VAE config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the VAE checkpoint")
    parser.add_argument("--folder-edente", type=str, required=True, help="Folder containing edentulous images")
    parser.add_argument("--folder-dente", type=str, default=None, help="Optional folder containing dentulous images")
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
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode")
    return parser.parse_args()


def load_image_as_base64(image_path: str, *, max_size: int = 512) -> str:
    """Load an image and convert it to base64 (PNG) for Dash display.

    Args:
        image_path (str): Path to the image.
        max_size (int): Maximum size (width/height) for display.

    Returns:
        str: Base64 data URI, or empty string on failure.
    """
    try:
        img = Image.open(image_path)
        if img.mode != "L":
            img = img.convert("L")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Failed to load image {image_path}: {exc}")
        return ""


def _title(method: str, has_dente: bool) -> str:
    """Build plot title string.

    Args:
        method (str): Dimensionality reduction method.
        has_dente (bool): Whether a second group is present.

    Returns:
        str: Title.
    """
    if has_dente:
        return f"{method.upper()} (● dente, ○ edente)"
    return method.upper()


def create_figure(
    *,
    method: str,
    proj_edente: np.ndarray,
    ids_edente: list[str],
    paths_edente: list[str],
    proj_dente: np.ndarray | None,
    ids_dente: list[str] | None,
    paths_dente: list[str] | None,
    patient_to_color: dict[str, str],
) -> go.Figure:
    """Create the Plotly figure.

    Args:
        method (str): Method name (``umap`` or ``tsne``).
        proj_edente (np.ndarray): 2D projection for edente points.
        ids_edente (list[str]): Patient IDs for edente points.
        paths_edente (list[str]): Image paths for edente points.
        proj_dente (np.ndarray | None): Optional 2D projection for dente points.
        ids_dente (list[str] | None): Patient IDs for dente points.
        paths_dente (list[str] | None): Image paths for dente points.
        patient_to_color (dict[str, str]): Patient -> color mapping.

    Returns:
        go.Figure: Plotly figure.
    """
    fig = go.Figure()

    def add_group(
        *,
        name: str,
        proj: np.ndarray,
        ids: list[str],
        paths: list[str],
        marker_symbol: str,
    ) -> None:
        hover_text: list[str] = []
        customdata: list[dict[str, Any]] = []
        colors: list[str] = []

        for index, (patient_id, path) in enumerate(zip(ids, paths, strict=True)):
            filename = Path(path).name
            hover_text.append(f"Patient: {patient_id}<br>Group: {name}<br>Index: {index}<br>File: {filename}")
            customdata.append({"path": path, "patient": patient_id, "group": name, "index": index})
            colors.append(patient_to_color.get(patient_id, "#333333"))

        fig.add_trace(
            go.Scatter(
                x=proj[:, 0],
                y=proj[:, 1],
                mode="markers",
                name=name,
                marker={
                    "size": 9,
                    "symbol": marker_symbol,
                    "color": colors,
                    "opacity": 0.75,
                    "line": {"width": 0.5, "color": "white"},
                },
                hovertext=hover_text,
                hoverinfo="text",
                customdata=customdata,
            )
        )

    add_group(
        name="edente",
        proj=proj_edente,
        ids=ids_edente,
        paths=paths_edente,
        marker_symbol="circle-open",
    )

    if proj_dente is not None and ids_dente is not None and paths_dente is not None:
        add_group(
            name="dente",
            proj=proj_dente,
            ids=ids_dente,
            paths=paths_dente,
            marker_symbol="circle",
        )

    fig.update_layout(
        title={"text": _title(method, proj_dente is not None), "x": 0.5, "xanchor": "center", "font": {"size": 18}},
        xaxis_title="Dim 1",
        yaxis_title="Dim 2",
        template="plotly_white",
        hovermode="closest",
        clickmode="event",
        margin={"l": 50, "r": 20, "t": 80, "b": 50},
        autosize=True,
        showlegend=True,
    )
    return fig


def main() -> None:
    """Run the interactive Dash app."""
    args = parse_args()

    device = init_device_and_seed(seed=None, print_monai_config=False)
    config, vae = load_config_and_model(args.config_file, args.checkpoint, device)
    patch_size = tuple(config.autoencoder_train["patch_size"])
    transforms = build_preprocess_transform(patch_size)

    analyzer = LatentSpaceAnalyzer(vae, device, transforms)

    cache_dir = Path(config.run_dir) / "analysis" / "latents_cache"
    cache = LatentCache(cache_dir)

    paths_edente = list_image_paths(args.folder_edente, max_images=args.num_samples)
    latent_edente, ids_edente, paths_edente = cache.get_or_encode_batch(
        image_paths=paths_edente,
        analyzer=analyzer,
        checkpoint_path=args.checkpoint,
        patch_size=patch_size,
        group_name="edente",
        batch_size=DEFAULT_BATCH_SIZE,
        show_progress=True,
    )

    latent_dente: np.ndarray | None = None
    ids_dente: list[str] | None = None
    paths_dente: list[str] | None = None
    if args.folder_dente:
        paths_dente = list_image_paths(args.folder_dente, max_images=args.num_samples)
        latent_dente, ids_dente, paths_dente = cache.get_or_encode_batch(
            image_paths=paths_dente,
            analyzer=analyzer,
            checkpoint_path=args.checkpoint,
            patch_size=patch_size,
            group_name="dente",
            batch_size=DEFAULT_BATCH_SIZE,
            show_progress=True,
        )

    all_ids = ids_edente + (ids_dente if ids_dente else [])
    _, patient_to_color = analyzer.create_patient_colormap(all_ids)

    total_points = latent_edente.shape[0] + (latent_dente.shape[0] if latent_dente is not None else 0)
    min_perplexity = 5
    if total_points <= min_perplexity:
        min_perplexity = max(1, total_points - 1)
    max_perplexity = min(100, max(min_perplexity, total_points // 3))
    max_perplexity = min(max_perplexity, max(1, total_points - 1))
    perplexity_initial = max(min_perplexity, min(DEFAULT_TSNE_PERPLEXITY, max_perplexity))
    umap_min_neighbors = 2
    if latent_edente.shape[0] <= umap_min_neighbors:
        umap_min_neighbors = max(1, latent_edente.shape[0] - 1)
    umap_max_neighbors = max(umap_min_neighbors, min(200, latent_edente.shape[0] - 1))
    umap_neighbors_initial = max(umap_min_neighbors, min(DEFAULT_UMAP_N_NEIGHBORS, umap_max_neighbors))

    def compute_projection(
        *,
        method: str,
        umap_n_neighbors: int,
        umap_min_dist: float,
        tsne_perplexity: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        with limit_threadpools(1):
            if method == "umap":
                umap_n_neighbors = max(umap_min_neighbors, min(int(umap_n_neighbors), umap_max_neighbors))
                proj_ed, umap_model, pca = analyzer.reduce_dimensionality_umap(
                    latent_edente,
                    n_neighbors=umap_n_neighbors,
                    min_dist=float(umap_min_dist),
                    random_state=None,
                    pca_components=50,
                )
                if latent_dente is None:
                    return proj_ed, None

                latent_dente_pca = pca.transform(latent_dente)
                proj_den = umap_model.transform(latent_dente_pca)
                return proj_ed, proj_den

            combined = (
                np.concatenate([latent_edente, latent_dente], axis=0)
                if latent_dente is not None
                else latent_edente
            )
            proj_all = analyzer.reduce_dimensionality_tsne(
                combined,
                perplexity=int(tsne_perplexity),
                random_state=None,
                pca_components=50,
            )
            split_idx = latent_edente.shape[0]
            proj_ed = proj_all[:split_idx]
            proj_den = proj_all[split_idx:] if latent_dente is not None else None
            return proj_ed, proj_den

    method_selected = args.method
    proj_edente, proj_dente = compute_projection(
        method=method_selected,
        umap_n_neighbors=umap_neighbors_initial,
        umap_min_dist=DEFAULT_UMAP_MIN_DIST,
        tsne_perplexity=perplexity_initial,
    )

    initial_figure = create_figure(
        method=method_selected,
        proj_edente=proj_edente,
        ids_edente=ids_edente,
        paths_edente=paths_edente,
        proj_dente=proj_dente,
        ids_dente=ids_dente,
        paths_dente=paths_dente,
        patient_to_color=patient_to_color,
    )

    app = dash.Dash(__name__)
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body { margin: 0 !important; padding: 0 !important; overflow: hidden !important; }
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

    tsne_step = 1 if max_perplexity < 10 else 5

    umap_controls_style = {
        "marginBottom": "20px",
        "padding": "10px",
        "backgroundColor": "#f9f9f9",
        "borderRadius": "5px",
        "display": "block" if method_selected == "umap" else "none",
    }
    tsne_controls_style = {
        "marginBottom": "20px",
        "padding": "10px",
        "backgroundColor": "#f9f9f9",
        "borderRadius": "5px",
        "display": "block" if method_selected == "tsne" else "none",
    }

    app.layout = html.Div(
        [
            dcc.Store(id="base-figure", data=initial_figure.to_dict()),
            dcc.Store(id="selection-store", data=None),
            html.Div(
                [
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
                    html.Div(
                        [
                            html.H3("Parameters", style={"textAlign": "center", "marginBottom": "15px"}),
                            html.Div(
                                [
                                    html.Label("method", style={"fontWeight": "bold", "display": "block"}),
                                    html.Div(method_selected.upper(), style={"fontSize": "14px", "color": "#444"}),
                                ],
                                style={"marginBottom": "10px"},
                            ),
                            html.Div(
                                [
                                    html.Label("n_neighbors", style={"fontWeight": "bold", "display": "block"}),
                                    dcc.Slider(
                                        id="umap-n-neighbors",
                                        min=umap_min_neighbors,
                                        max=umap_max_neighbors,
                                        step=5 if umap_max_neighbors >= 10 else 1,
                                        value=umap_neighbors_initial,
                                        marks={
                                            umap_min_neighbors: str(umap_min_neighbors),
                                            umap_neighbors_initial: str(umap_neighbors_initial),
                                            umap_max_neighbors: str(umap_max_neighbors),
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                    html.Label("min_dist", style={"fontWeight": "bold", "display": "block"}),
                                    dcc.Slider(
                                        id="umap-min-dist",
                                        min=0.0,
                                        max=0.99,
                                        step=0.05,
                                        value=DEFAULT_UMAP_MIN_DIST,
                                        marks={0.0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 0.99: "0.99"},
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                                id="umap-controls",
                                style=umap_controls_style,
                            ),
                            html.Div(
                                [
                                    html.Label("perplexity", style={"fontWeight": "bold", "display": "block"}),
                                    dcc.Slider(
                                        id="tsne-perplexity",
                                        min=min_perplexity,
                                        max=max_perplexity,
                                        step=tsne_step,
                                        value=perplexity_initial,
                                        marks={
                                            min_perplexity: str(min_perplexity),
                                            perplexity_initial: str(perplexity_initial),
                                            max_perplexity: str(max_perplexity),
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True},
                                    ),
                                ],
                                id="tsne-controls",
                                style=tsne_controls_style,
                            ),
                            html.H3("Selected Image", style={"textAlign": "center", "marginTop": "30px"}),
                            html.Div(
                                id="image-info",
                                style={
                                    "textAlign": "center",
                                    "marginBottom": "10px",
                                    "fontSize": "14px",
                                    "padding": "8px",
                                    "backgroundColor": "#fafafa",
                                    "borderRadius": "6px",
                                },
                            ),
                            html.Div(id="image-container", style={"textAlign": "center", "padding": "20px"}),
                            html.Div(
                                id="distance-info",
                                style={
                                    "textAlign": "center",
                                    "marginTop": "10px",
                                    "fontSize": "13px",
                                    "color": "#444",
                                    "padding": "10px",
                                    "borderRadius": "6px",
                                    "backgroundColor": "#fafafa",
                                },
                            ),
                        ],
                        style={
                            "flex": "1",
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

    if method_selected == "umap":

        @app.callback(
            Output("base-figure", "data"),
            [Input("umap-n-neighbors", "value"), Input("umap-min-dist", "value")],
        )
        def update_base_figure_umap(umap_n_neighbors: int, umap_min_dist: float) -> dict[str, Any]:
            """Recompute UMAP projection when parameters change.

            Args:
                umap_n_neighbors (int): UMAP n_neighbors value.
                umap_min_dist (float): UMAP min_dist value.

            Returns:
                dict[str, Any]: Figure dictionary.
            """
            proj_ed, proj_den = compute_projection(
                method=method_selected,
                umap_n_neighbors=umap_n_neighbors,
                umap_min_dist=umap_min_dist,
                tsne_perplexity=perplexity_initial,
            )
            fig = create_figure(
                method=method_selected,
                proj_edente=proj_ed,
                ids_edente=ids_edente,
                paths_edente=paths_edente,
                proj_dente=proj_den,
                ids_dente=ids_dente,
                paths_dente=paths_dente,
                patient_to_color=patient_to_color,
            )
            return fig.to_dict()

    else:

        @app.callback(
            Output("base-figure", "data"),
            [Input("tsne-perplexity", "value")],
        )
        def update_base_figure_tsne(tsne_perplexity: int) -> dict[str, Any]:
            """Recompute t-SNE projection when parameters change.

            Args:
                tsne_perplexity (int): t-SNE perplexity value.

            Returns:
                dict[str, Any]: Figure dictionary.
            """
            proj_ed, proj_den = compute_projection(
                method=method_selected,
                umap_n_neighbors=DEFAULT_UMAP_N_NEIGHBORS,
                umap_min_dist=DEFAULT_UMAP_MIN_DIST,
                tsne_perplexity=tsne_perplexity,
            )
            fig = create_figure(
                method=method_selected,
                proj_edente=proj_ed,
                ids_edente=ids_edente,
                paths_edente=paths_edente,
                proj_dente=proj_den,
                ids_dente=ids_dente,
                paths_dente=paths_dente,
                patient_to_color=patient_to_color,
            )
            return fig.to_dict()

    @app.callback(
        Output("latent-plot", "figure"),
        [Input("base-figure", "data"), Input("selection-store", "data")],
    )
    def overlay_selection(base_fig_data: dict[str, Any] | None, selection_data: dict[str, Any] | None) -> go.Figure:
        """Overlay a selection line on top of the current base figure.

        Args:
            base_fig_data (dict[str, Any] | None): Base figure serialized as dict.
            selection_data (dict[str, Any] | None): Selected points store.

        Returns:
            go.Figure: Figure with overlay.
        """
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
    def display_click_image(
        click_data: dict[str, Any] | None,
        selection_data: dict[str, Any] | None,
    ) -> tuple[Any, Any, dict[str, Any] | None, Any]:
        """Render clicked image and update selection/distance blocks.

        Args:
            click_data (dict[str, Any] | None): Plotly click event.
            selection_data (dict[str, Any] | None): Previously stored selection.

        Returns:
            tuple[Any, Any, dict[str, Any] | None, Any]: Image component, info component, updated selection, distance.
        """
        if click_data is None:
            return html.Div("Click on a point to view the image"), "", selection_data, ""

        point = click_data["points"][0]
        customdata = point.get("customdata", {})
        if not customdata or not customdata.get("path"):
            return html.Div("No image path available"), "", selection_data, ""

        image_path = str(customdata["path"])
        patient = str(customdata.get("patient", "Unknown"))
        group = str(customdata.get("group", "Unknown"))
        index = int(customdata.get("index", -1))
        x_coord = float(point.get("x"))
        y_coord = float(point.get("y"))

        img_src = load_image_as_base64(image_path)
        if not img_src:
            return html.Div(f"Error loading image: {image_path}"), "", selection_data, ""

        image_div = html.Img(
            src=img_src,
            style={
                "maxWidth": "100%",
                "maxHeight": "500px",
                "border": "2px solid #ddd",
                "borderRadius": "5px",
            },
        )

        new_point = {
            "group": group,
            "index": index,
            "x": x_coord,
            "y": y_coord,
            "patient": patient,
            "path": image_path,
        }

        existing = selection_data["points"] if selection_data and selection_data.get("points") else []
        points: list[dict[str, Any]]
        if len(existing) == 0:
            points = [new_point]
            point_label = "P1"
        elif len(existing) == 1:
            points = [existing[0], new_point]
            point_label = "P2"
        else:
            points = [new_point]
            point_label = "P1"

        info = html.Div(
            html.P(
                [html.Span(f"{point_label}: ", style={"fontWeight": "bold"}), html.Span(Path(image_path).name)],
                style={"margin": "5px 0", "fontSize": "12px", "color": "#666"},
            )
        )

        distance_block: Any = ""
        if len(points) == 2:
            p1, p2 = points

            distance_val: float | None = None
            try:
                if p1["group"] == p2["group"]:
                    if p1["group"] == "edente":
                        distance_val = latent_distance(latent_edente[p1["index"]], latent_edente[p2["index"]])
                    elif latent_dente is not None:
                        distance_val = latent_distance(latent_dente[p1["index"]], latent_dente[p2["index"]])
                else:
                    if latent_dente is not None:
                        lat_a = latent_edente if p1["group"] == "edente" else latent_dente
                        lat_b = latent_edente if p2["group"] == "edente" else latent_dente
                        distance_val = latent_distance_cross(lat_a, p1["index"], lat_b, p2["index"])
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] Failed to compute latent distance: {exc}")

            projection_distance: float | None = None
            ratio_distance: float | None = None
            try:
                projection_distance = float(
                    np.linalg.norm(np.array([p1["x"], p1["y"]]) - np.array([p2["x"], p2["y"]]))
                )
                if distance_val is not None and projection_distance not in (None, 0):
                    ratio_distance = distance_val / projection_distance
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] Failed to compute projection distance: {exc}")

            distance_block = html.Div(
                [
                    html.P(f"Latent distance: {distance_val:.4f}" if distance_val is not None else "Latent distance: n/a"),
                    html.P(
                        f"Projection distance (2D): {projection_distance:.4f}"
                        if projection_distance is not None
                        else "Projection distance (2D): n/a"
                    ),
                    html.P(
                        f"Latent / projection ratio: {ratio_distance:.4f}"
                        if ratio_distance is not None
                        else "Latent / projection ratio: n/a"
                    ),
                ],
                style={"borderRadius": "6px", "backgroundColor": "#fafafa", "padding": "10px"},
            )

        return image_div, info, {"points": points}, distance_block

    print(f"Dash running on http://localhost:{DEFAULT_PORT} (host=0.0.0.0)")
    app.run(debug=args.debug, port=DEFAULT_PORT, host="0.0.0.0")


if __name__ == "__main__":
    main()
