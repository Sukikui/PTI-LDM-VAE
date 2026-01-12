from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from pti_ldm_vae_v2.tools.mask_metrics_utils import (
    compute_bbox,
    compute_dente_width,
    compute_edente_widths,
)

@dataclass(frozen=True)
class MaskMetricsConfig:
    """Configuration for computing mask-based attributes.

    Attributes:
        edente_dir: Directory containing edente (toothless) mask TIFFs.
        dente_dir: Directory containing dente (toothed) mask TIFFs.
        output_edente: Output path for the edente attributes JSON.
        output_dente: Output path for the dente attributes JSON.
        pixel_size_mm: Pixel size in millimeters (used to convert mm heights to pixel offsets).
        dente_heights_mm: Heights (mm) from the bottom where dente widths are measured.
        edente_width_samples: Number of evenly spaced widths to sample for edente masks.
    """

    edente_dir: Path
    dente_dir: Path
    output_edente: Path
    output_dente: Path
    pixel_size_mm: float
    dente_heights_mm: tuple[float, ...]
    edente_width_samples: int


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a TIFF mask and convert it to a binary array.

    Args:
        path (Path): Path to a TIFF mask.

    Returns:
        np.ndarray: Binary mask with values in {0, 1}.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    return (mask > 0).astype(np.uint8)


def list_tif_files(path: Path) -> dict[str, Path]:
    """List TIFF files in a directory and map stems to paths.

    Args:
        path (Path): Directory containing TIFF files.

    Returns:
        dict[str, Path]: Mapping {stem: full_path}.
    """
    return {p.stem: p for p in sorted(path.iterdir()) if p.suffix.lower() in {".tif", ".tiff"}}


def pixel_offsets_mm(heights_mm: Sequence[float], pixel_size_mm: float) -> list[int]:
    """Convert physical offsets (mm) to pixel offsets.

    Args:
        heights_mm (Sequence[float]): Heights in millimeters.
        pixel_size_mm (float): Pixel size in millimeters.

    Returns:
        list[int]: Pixel offsets.
    """
    return [int(round(height / pixel_size_mm)) for height in heights_mm]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compute edente/dente mask attributes.")
    parser.add_argument(
        "--edente-dir",
        type=Path,
        default=Path("./data/edente"),
        help="Directory containing edente (toothless) masks (default: ./data/edente).",
    )
    parser.add_argument(
        "--dente-dir",
        type=Path,
        default=Path("./data/dente"),
        help="Directory containing dente (toothed) masks sampled at fixed heights (default: ./data/dente).",
    )
    parser.add_argument(
        "--output-edente",
        type=Path,
        default=Path("./data/metrics/attributes_edente.json"),
        help="Output JSON file for edente attributes (default: ./data/metrics/attributes_edente.json).",
    )
    parser.add_argument(
        "--output-dente",
        type=Path,
        default=Path("./data/metrics/attributes_dente.json"),
        help="Output JSON file for dente attributes (default: ./data/metrics/attributes_dente.json).",
    )
    parser.add_argument(
        "--pixel-size-mm",
        type=float,
        default=0.15,
        help="Pixel size (mm) used to convert dente heights to pixel offsets (default: 0.15).",
    )
    parser.add_argument(
        "--dente-heights-mm",
        type=float,
        nargs="+",
        default=(5.0, 10.0, 14.0, 18.0, 22.0),
        help=(
            "Heights (mm) from the bottom of the dente mask where widths are measured "
            "(default: 5 10 14 18 22)."
        ),
    )
    parser.add_argument(
        "--edente-width-samples",
        type=int,
        default=5,
        help="Number of evenly spaced widths for edente masks (default: 5).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MaskMetricsConfig:
    """Create a MaskMetricsConfig from CLI arguments only.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        MaskMetricsConfig: Normalized configuration.
    """
    return MaskMetricsConfig(
        edente_dir=args.edente_dir.expanduser().resolve(),
        dente_dir=args.dente_dir.expanduser().resolve(),
        output_edente=args.output_edente.expanduser().resolve(),
        output_dente=args.output_dente.expanduser().resolve(),
        pixel_size_mm=float(args.pixel_size_mm),
        dente_heights_mm=tuple(float(val) for val in args.dente_heights_mm),
        edente_width_samples=int(args.edente_width_samples),
    )


def prepare_paths(config: MaskMetricsConfig) -> tuple[dict[str, Path], dict[str, Path]]:
    """Validate directories and ensure output parents exist.

    Args:
        config (MaskMetricsConfig): Tool configuration.

    Returns:
        tuple[dict[str, Path], dict[str, Path]]: (edente_files, dente_files) mapped by stem.
    """
    if not config.edente_dir.is_dir():
        raise FileNotFoundError(f"Edente directory not found: {config.edente_dir}")
    if not config.dente_dir.is_dir():
        raise FileNotFoundError(f"Dente directory not found: {config.dente_dir}")

    config.output_edente.parent.mkdir(parents=True, exist_ok=True)
    config.output_dente.parent.mkdir(parents=True, exist_ok=True)

    return list_tif_files(config.edente_dir), list_tif_files(config.dente_dir)


def process_dataset(config: MaskMetricsConfig) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Compute attributes for all matching edente/dente pairs.

    Args:
        config (MaskMetricsConfig): Tool configuration.

    Returns:
        tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]: (attributes_edente, attributes_dente).
    """
    edente_files, dente_files = prepare_paths(config)
    common_stems = sorted(set(edente_files) & set(dente_files))
    if not common_stems:
        raise FileNotFoundError("No matching TIFF files found in both edente and dente directories.")

    dente_offsets = pixel_offsets_mm(config.dente_heights_mm, config.pixel_size_mm)
    attributes_edente: dict[str, dict[str, int]] = {}
    attributes_dente: dict[str, dict[str, int]] = {}
    skipped: list[tuple[str, str]] = []

    for stem in tqdm(common_stems, desc="Processing mask pairs", ncols=100):
        try:
            ed_mask = load_binary_mask(edente_files[stem])
            de_mask = load_binary_mask(dente_files[stem])

            x_min, y_min, bbox_w, bbox_h = compute_bbox(ed_mask)
            bbox_height_px, widths_edente = compute_edente_widths(
                ed_mask,
                x=x_min,
                y=y_min,
                width=bbox_w,
                height=bbox_h,
                samples=config.edente_width_samples,
            )

            attrs_edente = {"height_0": int(bbox_height_px)}
            for idx, value in enumerate(widths_edente):
                attrs_edente[f"width_{idx}"] = int(value)
            attributes_edente[edente_files[stem].name] = attrs_edente

            mask_height = int(de_mask.shape[0])
            row_indices = [max(0, min(mask_height - 1, mask_height - 1 - offset)) for offset in dente_offsets]
            widths_dente = [compute_dente_width(de_mask, row) for row in row_indices]

            attrs_dente = {"height_0": int(bbox_height_px)}
            for idx, value in enumerate(widths_dente):
                attrs_dente[f"width_{idx}"] = int(value)
            attributes_dente[dente_files[stem].name] = attrs_dente
        except (FileNotFoundError, ValueError, cv2.error) as exc:  # pylint: disable=protected-access
            skipped.append((stem, str(exc)))
            tqdm.write(f"Skipping {stem}: {exc}")

    if skipped:
        tqdm.write(f"Skipped {len(skipped)} pairs due to errors.")

    return attributes_edente, attributes_dente


def save_json(data: dict[str, dict[str, int]], path: Path) -> None:
    """Persist computed attributes to disk as JSON.

    Args:
        data (dict[str, dict[str, int]]): Attributes mapping filename -> metrics.
        path (Path): Output JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    config = build_config(args)

    attributes_edente, attributes_dente = process_dataset(config)
    save_json(attributes_edente, config.output_edente)
    save_json(attributes_dente, config.output_dente)

    config_dict = asdict(config)
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in config_dict.items()},
        "generated": [str(config.output_edente), str(config.output_dente)],
        "edente_entries": len(attributes_edente),
        "dente_entries": len(attributes_dente),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
