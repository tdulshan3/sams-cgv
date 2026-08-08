"""Cell cleaning module: table line removal and bounding box trimming.

This module provides functions to erase leftover printed border lines from inside
cell crops and trim ink masks to their bounding boxes.
"""

from __future__ import annotations

import cv2
import numpy as np

from src import config


def remove_table_lines(cell_bgr: np.ndarray) -> np.ndarray:
    """Erase leftover printed border lines from inside the crop.

    Uses morphological kernels to detect long horizontal/vertical line segments
    located near the edges of the cell crop, as well as clearing artifacts in the
    outer 2-pixel frame. Replaces detected line pixels with the estimated paper
    background colour.

    Args:
        cell_bgr: BGR image crop of the cell (H, W, 3), uint8.

    Returns:
        Cleaned BGR cell crop with border lines replaced by paper colour.
    """
    if cell_bgr is None or cell_bgr.size == 0:
        return cell_bgr

    cleaned = cell_bgr.copy()
    h, w = cleaned.shape[:2]
    if h < 5 or w < 5:
        return cleaned

    # Estimate paper background colour (median of bright pixels)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    bright_mask = gray > 180
    if np.any(bright_mask):
        paper_bg = np.median(cleaned[bright_mask], axis=0).astype(np.uint8)
    else:
        paper_bg = np.array([255, 255, 255], dtype=np.uint8)

    # Invert grayscale for line detection (lines are dark on light paper)
    # Threshold dark elements
    _, dark_mask = cv2.threshold(gray, config.DARK_MAX, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines near top/bottom margins (outer 25% of height)
    h_len = max(w // 4, 10)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, h_kernel)

    # Restrict horizontal line removal to top and bottom margins
    margin_y = max(h // 4, 3)
    h_line_mask = np.zeros_like(dark_mask)
    h_line_mask[:margin_y, :] = h_lines[:margin_y, :]
    h_line_mask[h - margin_y :, :] = h_lines[h - margin_y :, :]

    # Detect vertical lines near left/right margins (outer 25% of width)
    v_len = max(h // 4, 10)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_lines = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, v_kernel)

    # Restrict vertical line removal to left and right margins
    margin_x = max(w // 4, 3)
    v_line_mask = np.zeros_like(dark_mask)
    v_line_mask[:, :margin_x] = v_lines[:, :margin_x]
    v_line_mask[:, w - margin_x :] = v_lines[:, w - margin_x :]

    # Combined border lines mask
    border_lines_mask = cv2.bitwise_or(h_line_mask, v_line_mask)

    # Dilate slightly to ensure full coverage of line borders
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    border_lines_mask = cv2.dilate(border_lines_mask, dilate_kernel)

    # Paint border line pixels with paper background colour
    cleaned[border_lines_mask > 0] = paper_bg

    # Clear outer 2-pixel frame directly to ensure no border pixels touch crop edges
    cleaned = clear_crop_frame(cleaned, frame_px=2, bg_color=paper_bg)

    return cleaned


def clear_crop_frame(cell_bgr: np.ndarray, frame_px: int = 2, bg_color: np.ndarray | None = None) -> np.ndarray:
    """Clear the outer frame pixels of a BGR cell crop by filling them with paper background.

    Args:
        cell_bgr: BGR image crop of the cell.
        frame_px: Thickness of outer border frame in pixels to clear.
        bg_color: BGR background color tuple or array (defaults to white [255, 255, 255]).

    Returns:
        BGR image with outer frame replaced by background color.
    """
    if cell_bgr is None or cell_bgr.size == 0 or frame_px <= 0:
        return cell_bgr

    result = cell_bgr.copy()
    h, w = result.shape[:2]

    if bg_color is None:
        bg_color = np.array([255, 255, 255], dtype=np.uint8)

    result[:frame_px, :] = bg_color
    result[h - frame_px :, :] = bg_color
    result[:, :frame_px] = bg_color
    result[:, w - frame_px :] = bg_color

    return result



def drop_edge_blobs(mask: np.ndarray, margin: int = 2) -> np.ndarray:
    """Drop any connected components that touch the outer margin of the mask.

    Args:
        mask: uint8 binary mask (ink=255, background=0).
        margin: Number of outer pixels defining the frame boundary.

    Returns:
        Filtered uint8 binary mask with edge-touching components removed.
    """
    if mask is None or mask.size == 0 or not np.any(mask):
        return mask

    result = mask.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)

    h, w = result.shape[:2]

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]

        # Check if the blob touches the outer margin frame
        if x < margin or y < margin or (x + bw) > (w - margin) or (y + bh) > (h - margin):
            result[labels == i] = 0

    return result


def trim_to_content(mask: np.ndarray, pad: int = config.CELL_PAD) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Tight crop around the ink content, plus the bbox used.

    Args:
        mask: uint8 binary mask (ink=255, background=0).
        pad: Padding in pixels to add around the content bounding box.

    Returns:
        tuple containing:
            - cropped_mask: Tight crop of the mask with padding.
            - bbox: Bounding box tuple (x, y, w, h) relative to original mask.
    """
    if mask is None or mask.size == 0:
        return mask, (0, 0, 0, 0)

    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        h, w = mask.shape[:2]
        return mask.copy(), (0, 0, w, h)

    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))

    h, w = mask.shape[:2]

    # Apply padding
    x_min_padded = max(0, x_min - pad)
    y_min_padded = max(0, y_min - pad)
    x_max_padded = min(w, x_max + 1 + pad)
    y_max_padded = min(h, y_max + 1 + pad)

    bbox_w = x_max_padded - x_min_padded
    bbox_h = y_max_padded - y_min_padded

    cropped_mask = mask[y_min_padded:y_max_padded, x_min_padded:x_max_padded].copy()
    bbox = (x_min_padded, y_min_padded, bbox_w, bbox_h)

    return cropped_mask, bbox
