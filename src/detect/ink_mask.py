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
        return ink_mask_saturation(cell_bgr)
    elif method == "hsv":
        # For now, hsv delegates to saturation (will be expanded in subsequent subtasks)
        return ink_mask_saturation(cell_bgr)
    else:
        # Fallback to saturation for now until other branches are added in next subtasks
        return ink_mask_saturation(cell_bgr)
