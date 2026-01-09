from __future__ import annotations

import argparse

import numpy as np
import torch
from dash import Dash, Input, Output, dcc, html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pti_ldm_vae_v2.vae_regression_common import init_device_and_seed

from .build import build_frozen_vae
from .config import load_config
from .data import build_ldm_inference_transform
from .noise import build_gradient_noise_mask, create_initial_latent, read_noise_init_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for noisy latent visualization.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize a noisy dentate latent channel.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to LDM JSON config.")
    parser.add_argument("--input-path", required=True, help="Path to a dentate TIF image.")
    return parser.parse_args()


def _normalize_pair(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    *,
    low: int = 2,
    high: int = 98,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize clean/noisy tensors to the same 0-1 range.

    Args:
        clean (torch.Tensor): Clean latent channel [H, W].
        noisy (torch.Tensor): Noisy latent channel [H, W].
        low (int): Lower percentile for scaling.
        high (int): Upper percentile for scaling.

    Returns:
        tuple[np.ndarray, np.ndarray]: Normalized clean/noisy arrays in [0, 1].
    """
    clean_np = clean.detach().cpu().numpy()
    noisy_np = noisy.detach().cpu().numpy()
    stacked = np.concatenate([clean_np.reshape(-1), noisy_np.reshape(-1)])
    if stacked.size == 0:
        return np.zeros_like(clean_np), np.zeros_like(noisy_np)
    min_val = np.percentile(stacked, low)
    max_val = np.percentile(stacked, high)
    scale = max(max_val - min_val, 1e-8)
    clean_norm = np.clip((clean_np - min_val) / scale, 0.0, 1.0)
    noisy_norm = np.clip((noisy_np - min_val) / scale, 0.0, 1.0)
    return clean_norm, noisy_norm


def _normalize_mask(mask: torch.Tensor) -> np.ndarray:
    """Normalize a noise mask to 0-1.

    Args:
        mask (torch.Tensor): Mask tensor [H, W].

    Returns:
        np.ndarray: Normalized mask in [0, 1].
    """
    mask_np = mask.detach().cpu().numpy()
    min_val = float(mask_np.min())
    max_val = float(mask_np.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(mask_np)
    return (mask_np - min_val) / (max_val - min_val)


def _build_figure(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    mask: torch.Tensor,
    *,
    channel_idx: int,
) -> go.Figure:
    """Build a Plotly figure for a single latent channel.

    Args:
        clean (torch.Tensor): Clean latent channel [H, W].
        noisy (torch.Tensor): Noisy latent channel [H, W].
        mask (torch.Tensor): Noise mask [H, W].
        channel_idx (int): Selected channel index.

    Returns:
        go.Figure: Plotly figure with clean/noisy/mask panels.
    """
    clean_norm, noisy_norm = _normalize_pair(clean, noisy)
    mask_norm = _normalize_mask(mask)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("clean", "noisy", "mask"))
    fig.add_trace(
        go.Heatmap(z=clean_norm, colorscale="gray", zmin=0.0, zmax=1.0, showscale=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=noisy_norm, colorscale="gray", zmin=0.0, zmax=1.0, showscale=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(z=mask_norm, colorscale="gray", zmin=0.0, zmax=1.0, showscale=False),
        row=1,
        col=3,
    )
    fig.update_layout(
        title=f"Latent channel {channel_idx}",
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
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
        z_noisy = create_initial_latent(
            z_cond,
            init_mode=str(noise_init["init_mode"]),
            noise_top=float(noise_init["noise_top"]),
            noise_bottom=float(noise_init["noise_bottom"]),
            noise_exponent=float(noise_init["noise_exponent"]),
            noise_direction=str(noise_init["noise_direction"]),
            noise_weight=float(noise_init["noise_weight"]),
        )

    mask = build_gradient_noise_mask(
        z_cond.shape[2],
        z_cond.shape[3],
        noise_top=float(noise_init["noise_top"]),
        noise_bottom=float(noise_init["noise_bottom"]),
        noise_exponent=float(noise_init["noise_exponent"]),
        direction=str(noise_init["noise_direction"]),
        device=z_cond.device,
        dtype=z_cond.dtype,
    )[0, 0]

    clean_latents = z_cond[0].detach().cpu()
    noisy_latents = z_noisy[0].detach().cpu()
    channel_options = [{"label": str(idx), "value": idx} for idx in range(clean_latents.shape[0])]

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H3("Noisy dentate latent visualization"),
            html.Div(f"Latent shape: {tuple(clean_latents.shape)}"),
            dcc.Dropdown(
                id="channel-select",
                options=channel_options,
                value=0,
                clearable=False,
                style={"width": "200px"},
            ),
            dcc.Graph(id="latent-figure"),
        ]
    )

    @app.callback(Output("latent-figure", "figure"), Input("channel-select", "value"))
    def update_figure(channel_idx: int) -> go.Figure:
        """Update the figure based on the selected channel.

        Args:
            channel_idx (int): Selected channel index.

        Returns:
            go.Figure: Updated Plotly figure.
        """
        clean_channel = clean_latents[channel_idx]
        noisy_channel = noisy_latents[channel_idx]
        return _build_figure(clean_channel, noisy_channel, mask, channel_idx=channel_idx)

    app.run(host="0.0.0.0", port=8050, debug=False)


if __name__ == "__main__":
    main()
