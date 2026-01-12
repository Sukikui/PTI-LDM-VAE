from __future__ import annotations

import cv2
import numpy as np

DEFAULT_PRED_THRESHOLD = 0.2


def binary_mask_from_prediction(image: np.ndarray, *, threshold: float = DEFAULT_PRED_THRESHOLD) -> np.ndarray:
    """Convert a predicted image to a cleaned binary mask.

    Args:
        image (np.ndarray): Prediction array with shape [H, W] or [1, H, W].
        threshold (float): Absolute threshold for foreground detection.

    Returns:
        np.ndarray: Binary mask with values in {0, 1}.
    """
    if image.ndim == 3:
        image = image[0]
    if image.ndim != 2:
        raise ValueError(f"Expected 2D prediction, got shape {image.shape}.")

    mask = (np.abs(image) > threshold).astype(np.uint8)
    if mask.max() == 0:
        return mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    largest = max(contours, key=cv2.contourArea)
    cleaned = np.zeros_like(mask)
    cv2.drawContours(cleaned, [largest], -1, color=1, thickness=-1)
    return cleaned


def compute_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Compute the bounding box of the mask foreground.

    Args:
        mask (np.ndarray): Binary mask with values in {0, 1}.

    Returns:
        tuple[int, int, int, int]: Bounding box (x_min, y_min, width, height).
    """
    ys, xs = np.where(mask == 1)
    if ys.size == 0 or xs.size == 0:
        raise ValueError("Mask does not contain any foreground pixels.")
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def compute_edente_widths(
    mask: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    samples: int,
) -> tuple[int, list[int]]:
    """Compute multiple width samples across the edente mask bounding box.

    Args:
        mask (np.ndarray): Binary mask with values in {0, 1}.
        x (int): Bounding box x_min.
        y (int): Bounding box y_min.
        width (int): Bounding box width.
        height (int): Bounding box height.
        samples (int): Number of widths to sample.

    Returns:
        tuple[int, list[int]]: (bbox_height_px, sampled_widths_px).
    """
    if samples <= 0:
        return height, []

    ys = np.linspace(0, height, samples + 2, dtype=int)[1:-1][::-1] + y
    widths: list[int] = []
    for yy in ys:
        row = mask[yy, x : x + width]
        white = np.where(row == 1)[0]
        widths.append(int(white[-1] - white[0] + 1) if white.size else 0)
    return height, widths


def compute_dente_width(mask: np.ndarray, row_index: int) -> int:
    """Compute the width of the dente mask along a specific row.

    Args:
        mask (np.ndarray): Binary mask with values in {0, 1}.
        row_index (int): Row index (0 = top).

    Returns:
        int: Width in pixels for that row.
    """
    row = mask[row_index]
    white = np.where(row == 1)[0]
    return int(white[-1] - white[0] + 1) if white.size else 0
