from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for viewing a TIF image.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="View a single .tif image.")
    parser.add_argument("--input-path", required=True, type=Path, help="Path to the .tif image.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional path to save a PNG preview.")
    return parser.parse_args()


def load_tif(path: Path) -> np.ndarray:
    """Load a TIF image and squeeze singleton dimensions.

    Args:
        path (Path): Path to the input TIF image.

    Returns:
        np.ndarray: Image array as float32.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = tifffile.imread(str(path)).astype(np.float32)
    if image.ndim == 3:
        if image.shape[0] == 1:
            image = image[0]
        elif image.shape[-1] == 1:
            image = image[..., 0]
        else:
            image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeezing, got shape {image.shape}")
    return image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize an image to 0-255 for display with black background.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Normalized uint8 image.
    """
    mask = image != 0
    if not np.any(mask):
        return np.zeros_like(image, dtype=np.uint8)

    values = image[mask]
    min_val = float(values.min())
    max_val = float(values.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)

    scaled = np.zeros_like(image, dtype=np.float32)
    scaled[mask] = (values - min_val) / (max_val - min_val)
    return (scaled * 255).astype(np.uint8)


def render_image(image: np.ndarray, *, output_path: Path | None) -> None:
    """Render an image to screen or save it as PNG.

    Args:
        image (np.ndarray): Image array in uint8.
        output_path (Path | None): Optional output PNG path.
    """
    if output_path is None:
        plt.imshow(image, cmap="gray", vmin=0, vmax=255)
        plt.axis("off")
        plt.show()
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image, cmap="gray", vmin=0, vmax=255)
    print(f"[INFO] Saved preview to {output_path}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    image = load_tif(args.input_path)
    normalized = normalize_image(image)
    render_image(normalized, output_path=args.output_path)


if __name__ == "__main__":
    main()
