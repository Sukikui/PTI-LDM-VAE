#!/usr/bin/env python
"""Compute geometric metrics from dental mask TIFF files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class MetricConfig:
    """Container for mask metrics computation parameters."""

    edente_dir: Path
    dente_dir: Path
    output_edente: Path
    output_dente: Path
    pixel_size_mm: float
    dente_heights_mm: tuple[float, ...]
    edente_width_samples: int


def load_binary_mask(path: Path) -> np.ndarray:
    """Load a TIFF mask and convert it to a binary array."""
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")
    return (mask > 0).astype(np.uint8)


def compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Compute bounding box coordinates for the largest connected component."""
    ys, xs = np.where(mask == 1)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask does not contain any foreground pixels")
    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max(), ys.max()
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def compute_edente_widths(
    mask: np.ndarray, x: int, y: int, width: int, height: int, samples: int
) -> tuple[int, list[int]]:
    """Compute multiple width samples across the edente mask bounding box."""
    if samples <= 0:
        return height, []

    ys = np.linspace(0, height, samples + 2, dtype=int)[1:-1][::-1] + y
    widths = []
    for yy in ys:
        row = mask[yy, x : x + width]
        white = np.where(row == 1)[0]
        widths.append(int(white[-1] - white[0] + 1) if white.size else 0)
    return height, widths


def compute_dente_width(mask: np.ndarray, row_index: int) -> int:
    """Compute the width of the dental mask along a specific row."""
    row = mask[row_index]
    white = np.where(row == 1)[0]
    return int(white[-1] - white[0] + 1) if white.size else 0


def list_tif_files(path: Path) -> dict[str, Path]:
    """List TIFF files available in a directory."""
    return {p.stem: p for p in sorted(path.iterdir()) if p.suffix.lower() in {".tif", ".tiff"}}


def pixel_offsets_mm(heights_mm: Sequence[float], pixel_size_mm: float) -> list[int]:
    """Convert physical offsets in millimeters to pixel offsets."""
    return [int(round(height / pixel_size_mm)) for height in heights_mm]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
        help="Pixel size in millimeters used to convert dente heights to pixels (default: 0.15).",
    )
    parser.add_argument(
        "--dente-heights-mm",
        type=float,
        nargs="+",
        default=(5.0, 10.0, 14.0, 18.0, 22.0),
        help=(
            "Heights in millimeters from the bottom of the dente mask where widths are measured."
            " Can be customized to match your protocol (default: 5,10,14,18,22)."
        ),
    )
    parser.add_argument(
        "--edente-width-samples",
        type=int,
        default=5,
        help="Number of evenly spaced widths computed between top and bottom of the edente mask (default: 5).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MetricConfig:
    """Create a MetricConfig from CLI arguments only."""
    return MetricConfig(
        edente_dir=args.edente_dir.expanduser().resolve(),
        dente_dir=args.dente_dir.expanduser().resolve(),
        output_edente=args.output_edente.expanduser().resolve(),
        output_dente=args.output_dente.expanduser().resolve(),
        pixel_size_mm=float(args.pixel_size_mm),
        dente_heights_mm=tuple(float(val) for val in args.dente_heights_mm),
        edente_width_samples=int(args.edente_width_samples),
    )


def prepare_paths(config: MetricConfig) -> tuple[dict[str, Path], dict[str, Path]]:
    """Validate directories and determine output files."""
    if not config.edente_dir.is_dir():
        raise FileNotFoundError(f"Edente directory not found: {config.edente_dir}")
    if not config.dente_dir.is_dir():
        raise FileNotFoundError(f"Dente directory not found: {config.dente_dir}")

    config.output_edente.parent.mkdir(parents=True, exist_ok=True)
    config.output_dente.parent.mkdir(parents=True, exist_ok=True)

    return list_tif_files(config.edente_dir), list_tif_files(config.dente_dir)


def process_dataset(config: MetricConfig) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]]]:
    """Process a dataset and compute attributes for edente and dente masks."""
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

            x_min, y_min, width, height = compute_bbox(ed_mask)
            bbox_height_px, widths_edente = compute_edente_widths(
                ed_mask, x_min, y_min, width, height, config.edente_width_samples
            )

            attrs_edente = {"height_0": int(bbox_height_px)}
            for idx, value in enumerate(widths_edente):
                attrs_edente[f"width_{idx}"] = int(value)
            attributes_edente[edente_files[stem].name] = attrs_edente

            mask_height = de_mask.shape[0]
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
        tqdm.write(f"⚠️  Skipped {len(skipped)} pairs due to errors. See details above.")

    return attributes_edente, attributes_dente


def save_json(data: dict[str, dict[str, int]], path: Path) -> None:
    """Persist computed attributes to disk as JSON."""
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def main() -> None:
    """CLI entry point for computing mask metrics."""
    args = parse_args()
    config = build_config(args)
    attributes_edente, attributes_dente = process_dataset(config)
    save_json(attributes_edente, config.output_edente)
    save_json(attributes_dente, config.output_dente)

    config_dict = asdict(config)
    config_dict["edente_dir"] = str(config_dict["edente_dir"])
    config_dict["dente_dir"] = str(config_dict["dente_dir"])
    config_dict["output_edente"] = str(config_dict["output_edente"])
    config_dict["output_dente"] = str(config_dict["output_dente"])

    summary = {
        "config": config_dict,
        "generated": [str(config.output_edente), str(config.output_dente)],
        "edente_entries": len(attributes_edente),
        "dente_entries": len(attributes_dente),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
