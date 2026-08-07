"""Tests for cell cleaning and ink segmentation (M6)."""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.detect.cell_clean import drop_edge_blobs, remove_table_lines, trim_to_content


def test_remove_table_lines():
    """Verify leftover table border lines near cell edges are erased to paper background."""
    # Create synthetic cell image (white background with dark border lines)
    cell = np.full((100, 200, 3), 240, dtype=np.uint8)

    # Draw dark border line along top edge
    cell[:4, :] = 30
    # Draw dark border line along left edge
    cell[:, :4] = 30

    # Draw ink stroke in center
    cell[40:60, 80:120] = [180, 50, 50]  # Blue stroke

    cleaned = remove_table_lines(cell)

    # Check top edge line was erased (should be close to background paper colour 240)
    assert np.mean(cleaned[:4, 50:150]) > 200
    # Check center stroke was preserved
    assert np.mean(cleaned[40:60, 80:120, 0]) < 200


def test_drop_edge_blobs():
    """Verify blobs touching the 2-pixel outer frame are dropped."""
    mask = np.zeros((50, 50), dtype=np.uint8)

    # Blob 1: touching top edge
    mask[0:5, 10:15] = 255
    # Blob 2: inside center
    mask[20:30, 20:30] = 255

    cleaned_mask = drop_edge_blobs(mask, margin=2)

    assert cleaned_mask[2, 12] == 0
    assert np.all(cleaned_mask[20:30, 20:30] == 255)


def test_trim_to_content():
    """Verify content trimming and bounding box calculation."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 40:80] = 255

    pad = 4
    cropped, bbox = trim_to_content(mask, pad=pad)

    x, y, w, h = bbox
    assert x == 40 - pad
    assert y == 30 - pad
    assert w == 40 + 2 * pad
    assert h == 40 + 2 * pad
    assert cropped.shape == (h, w)

