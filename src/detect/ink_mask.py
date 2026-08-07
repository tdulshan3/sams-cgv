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


def close_stroke_gaps(mask: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Perform small morphological closing to join broken pen strokes.

    Uses a small ellipse kernel (3x3) to bridge minor gaps without inflating overall ink area.

    Args:
        mask: uint8 binary mask (ink=255, background=0).
        ksize: Size of the closing structuring element.

    Returns:
        uint8 mask with closed stroke gaps.
    """
    if mask is None or mask.size == 0 or not np.any(mask):
        return mask

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)



def clean_ink_mask(mask: np.ndarray, min_area: int = config.MIN_BLOB_AREA) -> np.ndarray:
    """Clean ink mask by closing small stroke gaps and removing specks.

    Args:
        mask: uint8 binary mask (ink=255, background=0).
        min_area: Minimum connected component area.

    Returns:
        Cleaned uint8 mask.
    """
    if mask is None or mask.size == 0:
        return mask

    # Close small broken gaps within pen strokes
    closed = close_stroke_gaps(mask, ksize=3)

    # Drop small component specks below min_area
    cleaned = filter_min_area(closed, min_area=min_area)
    return cleaned



def compute_ink_ratio(mask: np.ndarray) -> float:
    """Compute ratio of ink pixels to total cell pixels.

    Args:
        mask: uint8 binary mask (ink=255, background=0).

    Returns:
        Float in range [0.0, 1.0].
    """
    if mask is None or mask.size == 0:
        return 0.0
    ink_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    return float(ink_pixels / total_pixels)


def count_components(mask: np.ndarray) -> int:
    """Count connected components in cleaned mask (excluding background).

    Args:
        mask: uint8 binary mask (ink=255, background=0).

    Returns:
        Integer count of blobs.
    """
    if mask is None or mask.size == 0 or not np.any(mask):
        return 0
    num_labels, _ = cv2.connectedComponents(mask, connectivity=8)
    return max(0, num_labels - 1)


def ink_features(mask: np.ndarray) -> dict:
    """Extract ink features for decision stage (M7).

    Args:
        mask: uint8 binary mask (ink=255, background=0).

    Returns:
        Dictionary containing ink_ratio, components, and placeholders for
        subsequent feature computations.
    """
    if mask is None or mask.size == 0:
        return {
            "ink_ratio": 0.0,
            "components": 0,
            "stroke_bbox": None,
            "aspect": 0.0,
            "stroke_length": 0,
            "filled_ratio": 0.0,
            "centroid_offset": 0.0,
        }

    ratio = compute_ink_ratio(mask)
    num_comps = count_components(mask)

    return {
        "ink_ratio": ratio,
        "components": num_comps,
        "stroke_bbox": None,
        "aspect": 0.0,
        "stroke_length": 0,
        "filled_ratio": 0.0,
        "centroid_offset": 0.0,
    }


def count_pen_colours(cells_bgr: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, int]:

    """Count pen colour usage across all signature cells on a sheet.

    Args:
        cells_bgr: List of cell BGR crop images.
        masks: List of corresponding uint8 ink masks (same order).

    Returns:
        Dictionary mapping colour name ('blue', 'black', 'red', 'green', 'other')
        to the count of cells using that pen colour.
    """
    counts = {"blue": 0, "black": 0, "red": 0, "green": 0, "other": 0}

    for cell, mask in zip(cells_bgr, masks):
        if cell is None or mask is None or not np.any(mask):
            continue
        # Only count colour if there is noticeable ink in the cell (> 10 ink pixels)
        if np.count_nonzero(mask) >= config.MIN_BLOB_AREA:
            colour = dominant_pen_colour(cell, mask)
            counts[colour] = counts.get(colour, 0) + 1

    return counts


def dominant_pen_colour(cell_bgr: np.ndarray, mask: np.ndarray) -> str:

    """Identify the dominant pen colour used in a cell crop.

    Maps ink pixel hues through config.PEN_HUE_RANGES. If saturation is low,
    returns 'black'.

    Args:
        cell_bgr: BGR cell crop (H, W, 3), uint8.
        mask: uint8 binary mask (ink=255, background=0).

    Returns:
        One of 'blue' | 'black' | 'red' | 'green' | 'other'.
    """
    if cell_bgr is None or mask is None or not np.any(mask):
        return "black"

    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    ink_pixels = hsv[mask > 0]

    if len(ink_pixels) == 0:
        return "black"

    mean_sat = float(np.mean(ink_pixels[:, 1]))

    # Low saturation means black pen ink
    if mean_sat < config.SAT_MIN:
        return "black"

    # Count ink pixels falling into each pen colour range
    hues = ink_pixels[:, 0]
    counts: dict[str, int] = {}

    for color, ranges in config.PEN_HUE_RANGES.items():
        color_count = 0
        for low, high in ranges:
            color_count += int(np.sum((hues >= low) & (hues <= high)))
        counts[color] = color_count

    max_color = max(counts, key=lambda k: counts[k])
    max_count = counts[max_color]

    # Require at least 30% of ink pixels to match a recognized colour range
    if max_count > 0 and max_count >= 0.3 * len(ink_pixels):
        return max_color

    return "other"


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






