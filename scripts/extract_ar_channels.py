import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from pti_ldm_vae.data import create_vae_inference_dataloader
from pti_ldm_vae.models import VAEModel
from pti_ldm_vae.utils.vae_loader import load_vae_config, load_vae_model
from pti_ldm_vae.utils.visualization import normalize_image_to_uint8


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for AR channel extraction.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Extract and visualize an AR-regularized latent channel.")
    parser.add_argument("-c", "--config-file", required=True, help="Path to AR-VAE config JSON.")
    parser.add_argument("--checkpoint", required=True, help="Path to VAE checkpoint.")
    parser.add_argument("--input-dir", required=True, help="Directory with input .tif images.")
    parser.add_argument("--output-dir", required=True, help="Output directory for saved channels and summary.")
    parser.add_argument(
        "--attribute",
        required=True,
        help="Attribute name as defined in regularized_attributes.attribute_latent_mapping.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Limit number of images to process.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference (default: 8).")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers (default: 4).")
    parser.add_argument(
        "--save-format",
        choices=["tif", "png"],
        default="tif",
        help="Output format for channel maps (default: tif).",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace, config_name: str) -> Path:
    """Resolve and create output directory.

    Args:
        args: Parsed CLI arguments.
        config_name: Stem of the config file (unused when output_dir is required).

    Returns:
        Path to the output directory.
    """
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_channel_map(
    channel_map: np.ndarray, out_path: Path, save_format: str, normalize: bool = True, squeeze: bool = True
) -> None:
    """Persist a single channel map to disk.

    Args:
        channel_map: Latent channel map.
        out_path: Base path (without suffix).
        save_format: Output format: ``tif`` or ``png``.
        normalize: Whether to normalize PNG for visualization.
        squeeze: Whether to squeeze singleton dimensions.
    """
    data = np.squeeze(channel_map) if squeeze else channel_map
    if save_format == "tif":
        tifffile.imwrite(out_path.with_suffix(".tif"), data.astype(np.float32))
        return

    png_arr = normalize_image_to_uint8(data) if normalize else data.astype(np.uint8)
    tifffile.imwrite(out_path.with_suffix(".png"), png_arr)


def resolve_attribute_channel(reg_attrs: dict[str, Any], attribute_name: str) -> int:
    """Resolve the latent channel index for a given attribute name.

    Args:
        reg_attrs: regularized_attributes block from config.
        attribute_name: Name of the attribute (key in attribute_latent_mapping).

    Returns:
        Channel index as int.
    """
    raw_mapping = reg_attrs.get("attribute_latent_mapping", {})
    mapping = {k: v for k, v in raw_mapping.items() if not str(k).startswith("_")}
    if attribute_name not in mapping:
        raise ValueError(f"Attribute '{attribute_name}' not found in attribute_latent_mapping.")
    return int(mapping[attribute_name]["latent_channel"])


def extract_channels(args: argparse.Namespace) -> None:
    """Run channel extraction for the requested attribute.

    Args:
        args: Parsed CLI arguments.

    Returns:
        None
    """
    config = load_vae_config(args.config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder: VAEModel = load_vae_model(config, args.checkpoint, device)
    autoencoder.eval()

    config_name = Path(args.config_file).stem
    output_dir = resolve_output_dir(args, config_name)
    summary_path = output_dir / "summary.json"

    reg_attrs = getattr(config, "regularized_attributes", None)
    if not reg_attrs:
        raise ValueError("regularized_attributes block is missing in config.")
    target_channel = resolve_attribute_channel(reg_attrs, args.attribute)

    patch_size = tuple(config.autoencoder_train["patch_size"])
    dataloader, image_paths = create_vae_inference_dataloader(
        input_dir=args.input_dir,
        patch_size=patch_size,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )

    records: list[dict[str, Any]] = []
    path_iter = iter(image_paths)
    idx_global = 0

    for batch in tqdm(dataloader, desc="Extracting channels"):
        paths_batch = [next(path_iter) for _ in range(batch.shape[0])]
        with torch.no_grad():
            images = batch.to(device)
            _, z_mu, _ = autoencoder(images)

            if z_mu.dim() == 4:
                z_latent = z_mu
            elif z_mu.dim() == 2:
                z_latent = z_mu[:, :, None, None]
            else:
                raise ValueError(f"Unexpected latent shape: {z_mu.shape}")

            channel_maps = z_latent[:, target_channel]

        channel_maps_np = channel_maps.cpu().numpy()

        for cm, path in zip(channel_maps_np, paths_batch, strict=True):
            base = os.path.basename(path)
            stem = Path(base).stem

            out_base = output_dir / stem
            save_channel_map(cm, out_base, save_format=args.save_format)

            records.append(
                {
                    "index": idx_global,
                    "file": path,
                    "output_file": str(out_base.with_suffix(f".{args.save_format}")),
                    "attribute": args.attribute,
                    "latent_channel": target_channel,
                    "channel_mean": float(np.mean(cm)),
                    "channel_min": float(np.min(cm)),
                    "channel_max": float(np.max(cm)),
                }
            )
            idx_global += 1

    summary = {
        "config_file": args.config_file,
        "checkpoint": args.checkpoint,
        "attribute": args.attribute,
        "latent_channel": target_channel,
        "num_records": len(records),
        "stats": {
            "channel_mean_mean": float(np.mean([r["channel_mean"] for r in records])) if records else None,
            "channel_mean_std": float(np.std([r["channel_mean"] for r in records])) if records else None,
        },
        "records": records,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[INFO] Saved {len(records)} channel maps to {channels_dir}")
    print(f"[INFO] Summary written to {summary_path}")


def main() -> None:
    """Entry point.

    Returns:
        None
    """
    args = parse_args()
    extract_channels(args)


if __name__ == "__main__":
    main()
