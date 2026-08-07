"""Ink segmentation module for multi-colour pens.

Segments pen ink from paper background and printed table lines across multiple
colour spaces (HSV, LAB) supporting blue, black, red, green and other pens.
"""

from __future__ import annotations

import cv2
import numpy as np

from src import config


def ink_mask_saturation(cell_bgr: np.ndarray, sat_min: int = config.SAT_MIN) -> np.ndarray:
    """Extract ink mask using HSV saturation thresholding.

    Paper has near-zero saturation while coloured pens (blue, green, red) have high saturation.

    Args:
        cell_bgr: BGR cell crop (H, W, 3), uint8.
        sat_min: Minimum saturation threshold for coloured ink pixels.

    Returns:
        uint8 mask with ink = 255 and background = 0.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Create binary mask where saturation >= SAT_MIN
    mask = np.zeros_like(saturation, dtype=np.uint8)
    mask[saturation >= sat_min] = 255

    return mask


def ink_mask_darkness(cell_bgr: np.ndarray, val_max: int = config.VAL_MAX) -> np.ndarray:
    """Extract ink mask using HSV Value (darkness) thresholding for black pens.

    Black pen ink has low saturation and low value (darkness), whereas white paper
    has high value (~255).

    Args:
        cell_bgr: BGR cell crop (H, W, 3), uint8.
        val_max: Maximum value threshold for dark ink pixels.

    Returns:
        uint8 mask with ink = 255 and background = 0.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    # Create binary mask where value <= VAL_MAX
    mask = np.zeros_like(value, dtype=np.uint8)
    mask[value <= val_max] = 255

    return mask


def ink_mask_lab(
    cell_bgr: np.ndarray,
    chrom_min: float = 12.0,
    val_max: int = config.VAL_MAX,
) -> np.ndarray:
    """Extract ink mask using CIELAB colour space (a* and b* channels).

    Paper is neutral gray (a*, b* near 128). Coloured pens depart from 128 in a* or b*,
    and black pen ink has low L* (luminance).

    Args:
        cell_bgr: BGR cell crop (H, W, 3), uint8.
        chrom_min: Minimum chrominance distance from neutral (128, 128) for coloured ink.
        val_max: Maximum L* threshold for dark ink.

    Returns:
        uint8 mask with ink = 255 and background = 0.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    lab = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2LAB)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1].astype(np.float32) - 128.0
    b_chan = lab[:, :, 2].astype(np.float32) - 128.0

    chroma = np.sqrt(a_chan**2 + b_chan**2)

    # Combined mask: high chrominance (colour) or low luminance (dark ink)
    mask = np.zeros_like(l_chan, dtype=np.uint8)
    mask[(chroma >= chrom_min) | (l_chan <= val_max)] = 255

    return mask


def ink_mask_combined(
    cell_bgr: np.ndarray,
    sat_min: int = config.SAT_MIN,
    val_max: int = config.VAL_MAX,
) -> np.ndarray:
    """Extract ink mask combining HSV saturation (coloured pens) and value (black pens).

    Combines: mask = (saturation >= SAT_MIN) OR (value <= VAL_MAX).

    Args:
        cell_bgr: BGR cell crop (H, W, 3), uint8.
        sat_min: Minimum saturation threshold for coloured pen ink.
        val_max: Maximum value threshold for dark black pen ink.

    Returns:
        uint8 mask with ink = 255 and paper background = 0.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = np.zeros_like(saturation, dtype=np.uint8)
    mask[(saturation >= sat_min) | (value <= val_max)] = 255

    return mask


def filter_min_area(mask: np.ndarray, min_area: int = config.MIN_BLOB_AREA) -> np.ndarray:
    """Remove connected components smaller than min_area pixels.

    Args:
        mask: uint8 binary mask (ink=255, background=0).
        min_area: Minimum area in pixels required to keep a component.

    Returns:
        uint8 mask with small specks removed.
    """
    if mask is None or mask.size == 0 or not np.any(mask):
        return mask

    result = mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            result[labels == i] = 0

    return result


def clean_ink_mask(mask: np.ndarray, min_area: int = config.MIN_BLOB_AREA) -> np.ndarray:
    """Clean ink mask by removing small specks and edge-touching artifacts.

    Args:
        mask: uint8 binary mask (ink=255, background=0).
        min_area: Minimum connected component area.

    Returns:
        Cleaned uint8 mask.
    """
    if mask is None or mask.size == 0:
        return mask

    # Drop small component specks below min_area
    cleaned = filter_min_area(mask, min_area=min_area)
    return cleaned


def ink_mask(cell_bgr: np.ndarray, method: str = config.INK_METHOD) -> np.ndarray:
    """Segment ink pixels from a BGR cell crop.

    Args:
        cell_bgr: BGR cell crop (H, W, 3), uint8.
        method: Masking method: 'hsv' | 'lab' | 'saturation' | 'darkness' | 'combined'.

    Returns:
        uint8 binary mask with ink = 255 and paper background = 0.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    method = method.lower()

    if method == "saturation":
        raw_mask = ink_mask_saturation(cell_bgr)
    elif method == "darkness":
        raw_mask = ink_mask_darkness(cell_bgr)
    elif method == "lab":
        raw_mask = ink_mask_lab(cell_bgr)
    elif method in ("combined", "hsv"):
        raw_mask = ink_mask_combined(cell_bgr)
    else:
        raw_mask = ink_mask_combined(cell_bgr)

    return clean_ink_mask(raw_mask)




