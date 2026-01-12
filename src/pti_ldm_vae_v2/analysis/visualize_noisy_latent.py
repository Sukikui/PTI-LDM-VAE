from __future__ import annotations

import argparse

import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

from pti_ldm_vae_v2.common import init_device_and_seed
from pti_ldm_vae_v2.ldm.build import build_frozen_vae
from pti_ldm_vae_v2.ldm.config import load_config
from pti_ldm_vae_v2.ldm.data import build_ldm_inference_transform
from pti_ldm_vae_v2.ldm.noise import build_gradient_noise_mask, create_initial_latent, read_noise_init_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for noisy latent visualization.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize a noisy dentate latent channel.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to LDM JSON config.")
    parser.add_argument("--input-path", required=True, help="Path to a dentate TIF image.")
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Override latent scale factor used to match LDM training.",
    )
    return parser.parse_args()


def _compute_range(image: np.ndarray) -> tuple[float, float]:
    """Compute min/max range for a single image.

    Args:
        image (np.ndarray): Image to display.

    Returns:
        tuple[float, float]: (vmin, vmax) for the color scale.
    """
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(finite.min())
    vmax = float(finite.max())
    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    return vmin, vmax


def _build_figure(
    clean_channel: np.ndarray,
    noisy_channel: np.ndarray,
    noise_mask: np.ndarray,
    input_image: np.ndarray,
    decoded_noisy: np.ndarray,
    *,
    channel_idx: int,
) -> go.Figure:
    """Build a Plotly figure for a single latent channel.

    Args:
        clean_channel (np.ndarray): Selected clean latent channel [H, W].
        noisy_channel (np.ndarray): Selected noisy latent channel [H, W].
        noise_mask (np.ndarray): Noise mask applied to the latent [H, W].
        input_image (np.ndarray): Input dentate image [H, W].
        decoded_noisy (np.ndarray): Decoded noisy latent [H, W].
        channel_idx (int): Selected channel index.

    Returns:
        go.Figure: Plotly figure with latent/texture panels.
    """
    clean_vmin, clean_vmax = _compute_range(clean_channel)
    noisy_vmin, noisy_vmax = _compute_range(noisy_channel)
    mask_vmin, mask_vmax = _compute_range(noise_mask)
    input_vmin, input_vmax = _compute_range(input_image)
    decoded_vmin, decoded_vmax = _compute_range(decoded_noisy)
    h_spacing = 0.06
    col_width = (1.0 - h_spacing * 2) / 3
    col_rights = (col_width, col_width * 2 + h_spacing, col_width * 3 + h_spacing * 2)
    col_x = tuple(right + 0.01 for right in col_rights)
    row_y = (0.78, 0.22)
    bar_len = 0.32
    bar_thickness = 12

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{}, {}, {}], [{}, {}, None]],
        subplot_titles=(
            f"latent_clean (ch {channel_idx})",
            f"latent_noisy (ch {channel_idx})",
            "noise_mask",
            "input_dente",
            "decoded_noisy",
            "",
        ),
        horizontal_spacing=h_spacing,
        vertical_spacing=0.15,
    )
    fig.add_trace(
        go.Heatmap(
            z=clean_channel,
            colorscale="gray",
            zmin=clean_vmin,
            zmax=clean_vmax,
            showscale=True,
            colorbar=dict(
                title="clean",
                len=bar_len,
                y=row_y[0],
                x=col_x[0],
                xanchor="left",
                thickness=bar_thickness,
                thicknessmode="pixels",
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=noisy_channel,
            colorscale="gray",
            zmin=noisy_vmin,
            zmax=noisy_vmax,
            showscale=True,
            colorbar=dict(
                title="noisy",
                len=bar_len,
                y=row_y[0],
                x=col_x[1],
                xanchor="left",
                thickness=bar_thickness,
                thicknessmode="pixels",
            ),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(
            z=noise_mask,
            colorscale="gray",
            zmin=mask_vmin,
            zmax=mask_vmax,
            showscale=True,
            colorbar=dict(
                title="mask",
                len=bar_len,
                y=row_y[0],
                x=col_x[2],
                xanchor="left",
                thickness=bar_thickness,
                thicknessmode="pixels",
            ),
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Heatmap(
            z=input_image,
            colorscale="gray",
            zmin=input_vmin,
            zmax=input_vmax,
            showscale=True,
            colorbar=dict(
                title="input",
                len=bar_len,
                y=row_y[1],
                x=col_x[0],
                xanchor="left",
                thickness=bar_thickness,
                thicknessmode="pixels",
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=decoded_noisy,
            colorscale="gray",
            zmin=decoded_vmin,
            zmax=decoded_vmax,
            showscale=True,
            colorbar=dict(
                title="decoded",
                len=bar_len,
                y=row_y[1],
                x=col_x[1],
                xanchor="left",
                thickness=bar_thickness,
                thicknessmode="pixels",
            ),
        ),
        row=2,
        col=2,
    )
    fig.update_layout(
        title="LDM input visualization (sampling step 0)",
        margin=dict(l=20, r=90, t=40, b=20),
        height=780,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def main() -> None:
    """Entry point for noisy latent visualization."""
    args = parse_args()
    config = load_config(args.config_file)
    noise_init = read_noise_init_config(config)

    device = init_device_and_seed(config.get("seed"), print_monai_config=False)
    vae, _ = build_frozen_vae(
        config_file=config["vae"]["config_file"],
        checkpoint=config["vae"]["checkpoint"],
        device=device,
    )

    transform = build_ldm_inference_transform()
    image = transform(str(args.input_path))
    batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        z_cond = vae.encode_deterministic(batch)
        if args.scale_factor is None:
            latent_std = torch.std(z_cond).item()
            scale_factor = 1.0 / latent_std if latent_std > 0 else 1.0
        else:
            if args.scale_factor <= 0:
                raise ValueError("scale_factor must be positive.")
            scale_factor = float(args.scale_factor)
        z_scaled = z_cond * scale_factor
        init_mode = str(noise_init["init_mode"])
        noise_weight = float(noise_init["noise_weight"])
        z_noisy = create_initial_latent(
            z_scaled,
            init_mode=init_mode,
            noise_top=float(noise_init["noise_top"]),
            noise_bottom=float(noise_init["noise_bottom"]),
            noise_exponent=float(noise_init["noise_exponent"]),
            noise_direction=str(noise_init["noise_direction"]),
            noise_weight=noise_weight,
        )
        decoded_noisy = vae.decode_stage_2_outputs(z_noisy / scale_factor)
        if init_mode.strip().lower() in {"dentate_noisy", "noisy_dentate", "dentate"}:
            mask = build_gradient_noise_mask(
                z_scaled.shape[2],
                z_scaled.shape[3],
                noise_top=float(noise_init["noise_top"]),
                noise_bottom=float(noise_init["noise_bottom"]),
                noise_exponent=float(noise_init["noise_exponent"]),
                direction=str(noise_init["noise_direction"]),
                device=z_scaled.device,
                dtype=z_scaled.dtype,
            )
            mask = mask * noise_weight
        else:
            mask = torch.full(
                (1, 1, z_scaled.shape[2], z_scaled.shape[3]),
                noise_weight,
                device=z_scaled.device,
                dtype=z_scaled.dtype,
            )

    clean_latents = z_scaled[0].detach().cpu()
    noisy_latents = z_noisy[0].detach().cpu()
    noise_mask = mask[0, 0].detach().cpu().numpy()
    input_image = batch[0, 0].detach().cpu().numpy()
    decoded_noisy_image = decoded_noisy[0, 0].detach().cpu().numpy()
    channel_options = [{"label": str(idx), "value": idx} for idx in range(noisy_latents.shape[0])]

    app = Dash(__name__)
    page_style = {
        "minHeight": "100vh",
        "padding": "24px",
        "background": "linear-gradient(180deg, #f7f8fb 0%, #eef2f7 100%)",
        "fontFamily": '"Space Grotesk", "IBM Plex Sans", "Segoe UI", sans-serif',
        "color": "#111827",
    }
    panel_style = {
        "background": "#ffffff",
        "border": "1px solid #e5e7eb",
        "borderRadius": "16px",
        "boxShadow": "0 10px 30px rgba(15, 23, 42, 0.08)",
        "padding": "16px 18px",
    }
    header_row = {"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "12px"}
    muted_text = {"color": "#6b7280", "fontSize": "13px"}
    control_row = {"display": "flex", "alignItems": "center", "gap": "10px", "marginTop": "12px"}

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Noisy dentate latent visualization", style={"margin": "0"}),
                            html.Div(f"Latent shape: {tuple(noisy_latents.shape)}", style=muted_text),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Span("Channel", style={"fontSize": "12px", "color": "#6b7280"}),
                            dcc.Dropdown(
                                id="channel-select",
                                options=channel_options,
                                value=0,
                                clearable=False,
                                style={"width": "200px"},
                            ),
                        ],
                        style=control_row,
                    ),
                ],
                style={**panel_style, **header_row},
            ),
            html.Div(
                [dcc.Graph(id="latent-figure")],
                style={**panel_style, "marginTop": "16px"},
            ),
        ],
        style=page_style,
    )

    @app.callback(Output("latent-figure", "figure"), Input("channel-select", "value"))
    def update_figure(channel_idx: int) -> go.Figure:
        """Update the figure based on the selected channel.

        Args:
            channel_idx (int): Selected channel index.

        Returns:
            go.Figure: Updated Plotly figure.
        """
        clean_channel = clean_latents[channel_idx].detach().cpu().numpy()
        noisy_channel = noisy_latents[channel_idx].detach().cpu().numpy()
        return _build_figure(
            clean_channel,
            noisy_channel,
            noise_mask,
            input_image,
            decoded_noisy_image,
            channel_idx=channel_idx,
        )

    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
