import argparse
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from monai.transforms import Compose, EnsureChannelFirst, EnsureType, Resize

from pti_ldm_vae.analysis.common import TifReader
from pti_ldm_vae.data.transforms import LocalNormalizeByMask
from pti_ldm_vae.utils.vae_loader import load_vae_config, load_vae_model


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the AR channel viewer.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Interactive viewer for AR-VAE latent channels (single image).")
    parser.add_argument("-c", "--config-file", required=True, help="Path to AR-VAE config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--image-path", required=True, help="Path to the input .tif image.")
    parser.add_argument("--port", type=int, default=8052, help="Port for Dash server (default: 8052).")
    parser.add_argument("--host", default="127.0.0.1", help="Host for Dash server (default: 127.0.0.1).")
    parser.add_argument("--debug", action="store_true", help="Run Dash in debug mode.")
    return parser.parse_args()


def build_transform(patch_size: tuple[int, int]) -> Compose:
    """Create preprocessing transform matching VAE training pipeline.

    Args:
        patch_size: Spatial size (H, W) used to resize the input.

    Returns:
        Composed MONAI transform.
    """
    return Compose(
        [
            TifReader(),
            EnsureChannelFirst(channel_dim="no_channel"),
            Resize(patch_size),
            LocalNormalizeByMask(),
            EnsureType(dtype=torch.float32),
        ]
    )


def load_attribute_mapping(config: Any) -> dict[str, int]:
    """Extract attribute-to-channel mapping from config.

    Args:
        config: Parsed configuration namespace.

    Returns:
        Mapping from attribute name to latent channel index.

    Raises:
        ValueError: If regularized_attributes or attribute_latent_mapping is missing.
    """
    reg_attrs = getattr(config, "regularized_attributes", None)
    if not reg_attrs:
        raise ValueError("Config is missing regularized_attributes.")
    raw_mapping = reg_attrs.get("attribute_latent_mapping", {})
    mapping = {k: v for k, v in raw_mapping.items() if not str(k).startswith("_")}
    if not mapping:
        raise ValueError("attribute_latent_mapping is empty.")
    return {name: int(meta["latent_channel"]) for name, meta in mapping.items()}


def encode_image(
    image_path: str, autoencoder: torch.nn.Module, transform: Compose, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Encode a single image and return reconstructed output and latent means (deterministic).

    Args:
        image_path: Path to input image.
        autoencoder: Loaded VAE model.
        transform: Preprocessing transform.
        device: Torch device.

    Returns:
        Tuple of (reconstruction array, latent means array with shape [C, H, W]).
    """
    image = transform(image_path)
    batch = image.unsqueeze(0).to(device)
    with torch.no_grad():
        z_mu = autoencoder.encode_deterministic(batch)
        reconstruction = autoencoder.decode_stage_2_outputs(z_mu)

    recon_np = reconstruction.squeeze(0).cpu().numpy()

    if z_mu.dim() == 4:
        latents = z_mu[0]
    elif z_mu.dim() == 2:
        latents = z_mu[0][:, None, None]
    else:
        raise ValueError(f"Unexpected latent shape: {z_mu.shape}")

    return recon_np, latents.cpu().numpy()


def _normalize_to_unit(data: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] for visualization."""
    if data.size == 0:
        return data
    data_min, data_max = float(data.min()), float(data.max())
    if data_max <= data_min:
        return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min)


def _make_heatmap(
    data: np.ndarray,
    title: str,
    colorscale: str = "Viridis",
    showscale: bool = True,
    width: int = 340,
    height: int = 340,
    show_colorbar: bool | None = None,
) -> go.Figure:
    """Create a Plotly heatmap figure for a single-channel image with fixed sizing."""
    fig = go.Figure(
        data=go.Heatmap(
            z=np.squeeze(data),
            colorscale=colorscale,
            showscale=showscale if show_colorbar is None else show_colorbar,
            colorbar={"title": title, "len": 0.8} if (showscale if show_colorbar is None else show_colorbar) else None,
        )
    )
    fig.update_layout(
        title=title,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        height=height,
        width=width,
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(autorange="reversed")
    return fig


def build_app(
    original: np.ndarray,
    reconstruction: np.ndarray,
    latents: np.ndarray,
    attr_to_channel: dict[str, int],
    image_name: str,
) -> Dash:
    """Create the Dash application."""
    app = Dash(__name__)

    # Build dropdown options for all channels (regularized and unmapped)
    attr_by_idx = {idx: name for name, idx in attr_to_channel.items() if idx < latents.shape[0]}
    channel_options = []
    for idx in range(latents.shape[0]):
        if idx in attr_by_idx:
            label = f"{idx}: {attr_by_idx[idx]} (regularized)"
        else:
            label = f"{idx}: channel_{idx} (unmapped)"
        channel_options.append({"label": label, "value": idx})

    default_channel = channel_options[0]["value"] if channel_options else None

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2(f"AR-VAE Channels · {image_name}", style={"marginBottom": "6px"}),
                    html.P(
                        "Compare input vs reconstruction and browse every latent channel. Regularized attributes are highlighted.",
                        style={"color": "#475467", "fontSize": "14px", "marginTop": "0px"},
                    ),
                ],
                style={"padding": "16px 16px 0 16px", "background": "#f8fafc", "borderRadius": "10px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Input", style={"marginTop": "0"}),
                            dcc.Graph(
                                id="input-fig",
                                figure=_make_heatmap(
                                    _normalize_to_unit(original),
                                    "Input",
                                    colorscale="Gray",
                                    showscale=False,
                                    width=340,
                                    height=340,
                                ),
                                style={"height": "360px", "width": "360px"},
                            ),
                        ],
                        style={
                            "flex": "0 0 360px",
                            "background": "white",
                            "borderRadius": "12px",
                            "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
                            "padding": "12px",
                        },
                    ),
                    html.Div(
                        [
                            html.H4("Reconstruction", style={"marginTop": "0"}),
                            dcc.Graph(
                                id="recon-fig",
                                figure=_make_heatmap(
                                    _normalize_to_unit(reconstruction),
                                    "Reconstruction",
                                    colorscale="Gray",
                                    showscale=False,
                                    width=340,
                                    height=340,
                                ),
                                style={"height": "360px", "width": "360px"},
                            ),
                        ],
                        style={
                            "flex": "0 0 360px",
                            "background": "white",
                            "borderRadius": "12px",
                            "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
                            "padding": "12px",
                        },
                    ),
                    html.Div(
                        [
                            html.H4("Latent Channel", style={"marginTop": "0"}),
                            dcc.Dropdown(
                                id="channel-dropdown",
                                options=channel_options,
                                value=default_channel,
                                clearable=False,
                                style={"marginBottom": "10px"},
                            ),
                            dcc.Graph(id="channel-fig", style={"height": "380px", "width": "400px"}),
                        ],
                        style={
                            "flex": "0 0 420px",
                            "background": "white",
                            "borderRadius": "12px",
                            "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
                            "padding": "12px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "gap": "12px",
                    "padding": "16px",
                    "background": "#eef2f7",
                    "borderRadius": "12px",
                    "alignItems": "stretch",
                    "flexWrap": "nowrap",
                    "overflowX": "auto",
                },
            ),
            dcc.Store(id="latent-store", data=latents.tolist()),
            dcc.Store(id="attr-map", data=attr_to_channel),
        ],
        style={
            "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            "background": "#e6ecf2",
        },
    )

    @app.callback(
        Output("channel-fig", "figure"),
        Input("channel-dropdown", "value"),
        State("latent-store", "data"),
        State("attr-map", "data"),
    )
    def update_channel_fig(selected_channel: int | None, latent_data: list[list[list[float]]], mapping: dict[str, int]):
        """Update channel heatmap when dropdown changes."""
        if selected_channel is None:
            return _make_heatmap(np.zeros_like(latents[0]), "No channel")
        latent_array = np.array(latent_data)
        if selected_channel >= latent_array.shape[0]:
            return _make_heatmap(np.zeros_like(latent_array[0]), f"{selected_channel} (out of range)")
        channel_map = latent_array[selected_channel]

        attr_name = next((name for name, idx in mapping.items() if idx == selected_channel), None)
        title = f"ch {selected_channel}"
        if attr_name:
            title = f"ch {selected_channel}: {attr_name} (regularized)"
        else:
            title = f"ch {selected_channel}: unmapped"

        # Hide colorbar to keep plot width stable; title carries info
        return _make_heatmap(
            _normalize_to_unit(channel_map), title, showscale=False, show_colorbar=False, width=420, height=360
        )

    return app


def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_vae_config(args.config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    attr_to_channel = load_attribute_mapping(config)
    patch_size = tuple(config.autoencoder_train["patch_size"])
    transform = build_transform(patch_size)

    autoencoder = load_vae_model(config, args.checkpoint, device)
    autoencoder.eval()

    original = TifReader()(args.image_path)
    reconstruction, latents = encode_image(args.image_path, autoencoder, transform, device)

    app = build_app(original, reconstruction, latents, attr_to_channel, Path(args.image_path).name)
    app.run(debug=args.debug, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
