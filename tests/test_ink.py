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


def test_clear_crop_frame():
    """Verify outer 2-pixel border frame of BGR crop is cleared."""
    from src.detect.cell_clean import clear_crop_frame

    cell = np.zeros((40, 40, 3), dtype=np.uint8)
    cell[:2, :] = 100  # Dark edge artifact
    cell[:, :2] = 100

    cleared = clear_crop_frame(cell, frame_px=2)
    assert np.all(cleared[:2, :] == 255)
    assert np.all(cleared[:, :2] == 255)



def test_ink_mask_saturation():
    """Verify HSV saturation thresholding detects coloured pen strokes (blue, red, green)."""
    from src.detect.ink_mask import ink_mask, ink_mask_saturation

    # White cell (zero saturation)
    cell = np.full((50, 100, 3), 255, dtype=np.uint8)

    # Add a blue pen stroke (high saturation)
    cell[20:30, 40:60] = [255, 0, 0]  # Pure BGR blue

    mask = ink_mask_saturation(cell, sat_min=60)

    # Blue stroke region should be 255
    assert np.all(mask[20:30, 40:60] == 255)
    # Background should be 0
    assert np.all(mask[0:10, 0:10] == 0)

    # Check ink_mask with method='saturation'
    mask_method = ink_mask(cell, method="saturation")
    assert np.array_equal(mask, mask_method)


def test_ink_mask_darkness():
    """Verify darkness (value) thresholding branch detects black pen ink."""
    from src.detect.ink_mask import ink_mask, ink_mask_darkness

    # White cell paper (high value ~255)
    cell = np.full((50, 100, 3), 245, dtype=np.uint8)

    # Black pen stroke (low value ~40, low saturation)
    cell[15:35, 30:70] = [40, 40, 40]

    mask = ink_mask_darkness(cell, val_max=200)

    # Black stroke should be 255
    assert np.all(mask[15:35, 30:70] == 255)
    # Background paper should be 0
    assert np.all(mask[0:10, 0:10] == 0)

    # Check ink_mask with method='darkness'
    mask_method = ink_mask(cell, method="darkness")
    assert np.array_equal(mask, mask_method)


def test_ink_mask_lab():
    """Verify LAB colour space masking detects both coloured ink and dark ink."""
    from src.detect.ink_mask import ink_mask, ink_mask_lab

    # White cell paper
    cell = np.full((50, 100, 3), 245, dtype=np.uint8)

    # Add red pen stroke (high chrominance in LAB a* channel)
    cell[10:20, 20:40] = [30, 30, 220]  # Red in BGR

    # Add black pen stroke (low luminance L*)
    cell[30:40, 60:80] = [30, 30, 30]

    mask = ink_mask_lab(cell)

    assert np.all(mask[10:20, 20:40] == 255)
    assert np.all(mask[30:40, 60:80] == 255)
    assert np.all(mask[0:5, 0:5] == 0)

    # Check ink_mask with method='lab'
    mask_method = ink_mask(cell, method="lab")
    assert np.array_equal(mask, mask_method)


def test_ink_mask_combined():
    """Verify combined HSV (saturation OR value) catches blue, black and red ink strokes."""
    from src.detect.ink_mask import ink_mask, ink_mask_combined

    # White cell paper (high value, low saturation)
    cell = np.full((60, 120, 3), 245, dtype=np.uint8)

    # Blue stroke (high saturation)
    cell[10:20, 10:40] = [220, 50, 20]  # BGR blue

    # Black stroke (low value, low saturation)
    cell[25:35, 50:80] = [30, 30, 30]

    # Red stroke (high saturation)
    cell[40:50, 90:110] = [20, 20, 220]  # BGR red

    mask = ink_mask_combined(cell)

    assert np.all(mask[10:20, 10:40] == 255)
    assert np.all(mask[25:35, 50:80] == 255)
    assert np.all(mask[40:50, 90:110] == 255)
    assert np.all(mask[0:5, 0:5] == 0)

    # Check ink_mask default/combined
    mask_method = ink_mask(cell, method="combined")
    assert np.array_equal(mask, mask_method)


def test_filter_min_area():
    """Verify a speck of 4 pixels is removed by MIN_BLOB_AREA while a 200-pixel stroke is kept."""
    from src.detect.ink_mask import filter_min_area

    mask = np.zeros((100, 100), dtype=np.uint8)

    # 4-pixel speck (2x2)
    mask[10:12, 10:12] = 255

    # 200-pixel stroke (10x20)
    mask[30:40, 30:50] = 255

    cleaned = filter_min_area(mask, min_area=12)

    # 4-pixel speck should be removed (0)
    assert np.count_nonzero(cleaned[10:12, 10:12]) == 0
    # 200-pixel stroke should be kept (255)
    assert np.count_nonzero(cleaned[30:40, 30:50]) == 200


def test_close_stroke_gaps():
    """Verify small 1-2 pixel breaks inside a pen stroke are joined by morphological closing."""
    from src.detect.ink_mask import close_stroke_gaps

    mask = np.zeros((50, 100), dtype=np.uint8)

    # Stroke segment 1
    mask[20:25, 20:40] = 255
    # 1-pixel gap at col 40
    # Stroke segment 2
    mask[20:25, 41:60] = 255

    closed = close_stroke_gaps(mask, ksize=3)

    # The 1-pixel gap at col 40 should now be filled (255)
    assert np.all(closed[20:25, 40] == 255)


def test_dominant_pen_colour():
    """Verify dominant_pen_colour returns 'blue' for pure blue, 'red' for red, 'green' for green, 'black' for black."""
    from src.detect.ink_mask import dominant_pen_colour

    # 1. Blue stroke
    cell_blue = np.full((40, 60, 3), 245, dtype=np.uint8)
    cell_blue[10:30, 10:50] = [220, 50, 20]  # BGR blue (Hue ~105)
    mask_blue = np.zeros((40, 60), dtype=np.uint8)
    mask_blue[10:30, 10:50] = 255
    assert dominant_pen_colour(cell_blue, mask_blue) == "blue"

    # 2. Black stroke (low saturation)
    cell_black = np.full((40, 60, 3), 245, dtype=np.uint8)
    cell_black[10:30, 10:50] = [40, 40, 40]  # Black
    mask_black = np.zeros((40, 60), dtype=np.uint8)
    mask_black[10:30, 10:50] = 255
    assert dominant_pen_colour(cell_black, mask_black) == "black"

    # 3. Red stroke
    cell_red = np.full((40, 60, 3), 245, dtype=np.uint8)
    cell_red[10:30, 10:50] = [20, 20, 220]  # BGR red (Hue ~0/180)
    mask_red = np.zeros((40, 60), dtype=np.uint8)
    mask_red[10:30, 10:50] = 255
    assert dominant_pen_colour(cell_red, mask_red) == "red"

    # 4. Green stroke
    cell_green = np.full((40, 60, 3), 245, dtype=np.uint8)
    cell_green[10:30, 10:50] = [20, 200, 20]  # BGR green (Hue ~60)
    mask_green = np.zeros((40, 60), dtype=np.uint8)
    mask_green[10:30, 10:50] = 255
    assert dominant_pen_colour(cell_green, mask_green) == "green"


def test_count_pen_colours():
    """Verify count_pen_colours correctly tallies pen colour usage across multiple cells."""
    from src.detect.ink_mask import count_pen_colours

    cell_blue = np.full((40, 60, 3), 245, dtype=np.uint8)
    cell_blue[10:30, 10:50] = [220, 50, 20]
    mask_blue = np.zeros((40, 60), dtype=np.uint8)
    mask_blue[10:30, 10:50] = 255

    cell_black = np.full((40, 60, 3), 245, dtype=np.uint8)
    cell_black[10:30, 10:50] = [40, 40, 40]
    mask_black = np.zeros((40, 60), dtype=np.uint8)
    mask_black[10:30, 10:50] = 255

    cells = [cell_blue, cell_black, cell_blue]
    masks = [mask_blue, mask_black, mask_blue]

    counts = count_pen_colours(cells, masks)
    assert counts["blue"] == 2
    assert counts["black"] == 1
    assert counts["red"] == 0
    assert counts["green"] == 0


def test_ink_features_basic():
    """Verify synthetic blank cell gives ink_ratio == 0 and components == 0, and stroke cell gives ratio > 0."""
    from src.detect.ink_mask import ink_features

    # 1. Blank cell mask
    blank_mask = np.zeros((50, 100), dtype=np.uint8)
    feats_blank = ink_features(blank_mask)
    assert feats_blank["ink_ratio"] == 0.0
    assert feats_blank["components"] == 0

    # 2. Stroke cell mask (50x100 = 5000 pixels, stroke is 10x20 = 200 pixels)
    stroke_mask = np.zeros((50, 100), dtype=np.uint8)
    stroke_mask[20:30, 40:60] = 255
    feats_stroke = ink_features(stroke_mask)
    assert feats_stroke["ink_ratio"] == pytest.approx(200 / 5000)
    assert feats_stroke["components"] == 1


def test_ink_features_bbox_aspect():
    """Verify stroke_bbox and aspect ratio feature computations."""
    from src.detect.ink_mask import ink_features

    # Signature-like wide stroke: y from 10 to 30 (h=20), x from 10 to 90 (w=80)
    mask = np.zeros((60, 120), dtype=np.uint8)
    mask[10:30, 10:90] = 255

    feats = ink_features(mask)
    assert feats["stroke_bbox"] == (10, 10, 80, 20)
    assert feats["aspect"] == pytest.approx(80 / 20)  # 4.0


def test_ink_features_skeleton_fill():
    """Verify stroke_length, filled_ratio and centroid_offset feature extraction."""
    from src.detect.ink_mask import ink_features

    mask = np.zeros((100, 100), dtype=np.uint8)

    # 10x10 square of ink centered at (45..55, 45..55)
    mask[45:55, 45:55] = 255

    feats = ink_features(mask)
    assert feats["stroke_length"] > 0
    assert feats["filled_ratio"] == pytest.approx(1.0)  # Solid 10x10 block fills 100% of its bbox
    assert feats["centroid_offset"] == pytest.approx(0.5, abs=1.0)  # Centroid near center (50, 50)


def test_save_cell_outputs(tmp_path, monkeypatch):
    """Verify cell crop and mask are saved to disk in the correct folder layout."""
    from pathlib import Path

    from src import config
    from src.detect.ink_mask import save_cell_outputs

    monkeypatch.setattr(config, "CELLS", tmp_path / "cells")

    cell_bgr = np.full((30, 60, 3), 200, dtype=np.uint8)
    mask = np.full((30, 60), 255, dtype=np.uint8)

    crop_path_str, mask_path_str = save_cell_outputs("12.07.2019", 0, cell_bgr, mask)

    assert Path(crop_path_str).is_file()
    assert Path(mask_path_str).is_file()
    assert "12.07.2019" in crop_path_str
    assert "row_0.png" in crop_path_str
    assert "row_0_mask.png" in mask_path_str


def test_ink_stage():
    """Verify InkStage runs over context cell list, returns InkResults, and produces montage figures."""
    from pathlib import Path

    from src.detect.ink_mask import InkStage
    from src.models import Cell, SheetMeta


    stage = InkStage()
    assert stage.name == "ink"

    # Create dummy cell with BGR image
    cell_img = np.full((40, 80, 3), 240, dtype=np.uint8)
    cell_img[10:30, 20:60] = [220, 40, 20]  # Blue stroke

    cell = Cell(row=0, col=4, bbox=(10, 10, 80, 40), image=cell_img)
    sheet = SheetMeta(path=Path("data/sheets/12.07.2019.png"), date="12.07.2019")

    ctx = {"cells": [cell], "sheet": sheet}

    res_ctx = stage.run(ctx)

    assert "ink" in res_ctx
    ink_list = res_ctx["ink"]
    assert len(ink_list) == 1

    ink_res = ink_list[0]
    assert ink_res.ink_ratio > 0
    assert ink_res.components >= 1
    assert ink_res.mask is not None

    figs = stage.figures()
    assert "ink_segmentation" in figs
    assert isinstance(figs["ink_segmentation"], np.ndarray)


def test_faded_ink_detection():
    """Verify lower saturation threshold (SAT_MIN=50) detects faint/faded ink strokes."""
    from src.detect.ink_mask import ink_mask_saturation

    cell = np.full((40, 60, 3), 245, dtype=np.uint8)
    # Faded blue stroke with moderate saturation (~55)
    cell[10:20, 10:40] = [200, 150, 140]

    mask = ink_mask_saturation(cell, sat_min=50)
    assert np.all(mask[10:20, 10:40] == 255)














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

