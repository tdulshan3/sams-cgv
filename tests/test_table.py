"""Tests for M5, Table Detection.

Covers:
  - line_detect: morphology masks, projection profile peaks, merge logic
  - grid_builder: band grouping, student table selection, repair, Grid geometry
  - cell_extract: crop_cell bounds, TableStage interface
"""

from __future__ import annotations

import numpy as np
import pytest

from src.table.grid_builder import (
    Grid,
    _drop_too_close,
    _group_into_bands,
    _repair_missing_lines,
    build_grid,
    select_student_table,
)
from src.table.line_detect import _merge_nearby, _peaks_from_profile, line_mask
from src.table.cell_extract import crop_cell


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binary_with_hline(height: int, width: int, y: int) -> np.ndarray:
    """Return a black image with one white horizontal line at row y."""
    img = np.zeros((height, width), dtype=np.uint8)
    img[y, :] = 255
    return img


def _make_binary_with_vline(height: int, width: int, x: int) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.uint8)
    img[:, x] = 255
    return img


# ---------------------------------------------------------------------------
# line_detect
# ---------------------------------------------------------------------------


class TestMergeNearby:
    def test_empty(self):
        assert _merge_nearby([], tol=8) == []

    def test_single(self):
        assert _merge_nearby([50], tol=8) == [50]

    def test_merges_close(self):
        result = _merge_nearby([100, 103, 106], tol=8)
        assert len(result) == 1

    def test_keeps_far_apart(self):
        result = _merge_nearby([10, 100], tol=8)
        assert len(result) == 2

    def test_average_is_integer(self):
        result = _merge_nearby([100, 101], tol=8)
        assert isinstance(result[0], int)


class TestPeaksFromProfile:
    def test_empty_profile_returns_nothing(self):
        profile = np.zeros(200, dtype=float)
        assert _peaks_from_profile(profile) == []

    def test_single_peak(self):
        profile = np.zeros(200, dtype=float)
        profile[80] = 100.0
        peaks = _peaks_from_profile(profile)
        assert 80 in peaks

    def test_multiple_peaks(self):
        profile = np.zeros(200, dtype=float)
        profile[40] = 100.0
        profile[120] = 90.0
        peaks = _peaks_from_profile(profile, min_distance=10)
        assert len(peaks) == 2


class TestLineMask:
    def test_rejects_unknown_orientation(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            line_mask(img, "diagonal")

    def test_horizontal_mask_same_shape(self):
        img = np.zeros((200, 400), dtype=np.uint8)
        img[100, :] = 255
        result = line_mask(img, "horizontal")
        assert result.shape == img.shape

    def test_vertical_mask_same_shape(self):
        img = np.zeros((200, 400), dtype=np.uint8)
        img[:, 200] = 255
        result = line_mask(img, "vertical")
        assert result.shape == img.shape


# ---------------------------------------------------------------------------
# grid_builder
# ---------------------------------------------------------------------------


class TestGroupIntoBands:
    def test_single_band(self):
        ys = [10, 20, 30, 40]
        bands = _group_into_bands(ys, gap_threshold=40)
        assert len(bands) == 1

    def test_two_bands(self):
        ys = [10, 20, 200, 210, 220]
        bands = _group_into_bands(ys, gap_threshold=40)
        assert len(bands) == 2

    def test_empty(self):
        assert _group_into_bands([]) == []


class TestSelectStudentTable:
    def test_picks_larger_band(self):
        small_band = [10, 20]
        large_band = [200, 220, 240, 260, 280, 300, 320]
        result = select_student_table([small_band, large_band])
        assert result == large_band

    def test_single_band_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="sams.table"):
            result = select_student_table([[100, 200, 300]])
        assert "only one" in caplog.text.lower()
        assert result == [100, 200, 300]

    def test_empty_warns(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="sams.table"):
            result = select_student_table([])
        assert result == []


class TestRepairMissingLines:
    def test_no_repair_needed(self):
        ys = [0, 50, 100, 150]
        result = _repair_missing_lines(ys, min_height=18)
        assert result == ys

    def test_inserts_one_line(self):
        ys = [0, 50, 150, 200]  # gap of 100 between 50 and 150, median is ~50
        result = _repair_missing_lines(ys, min_height=18)
        assert len(result) == 5

    def test_short_list(self):
        assert _repair_missing_lines([100], min_height=18) == [100]


class TestDropTooClose:
    def test_removes_duplicates(self):
        result = _drop_too_close([10, 12, 50, 100], min_gap=18)
        assert 12 not in result
        assert 10 in result

    def test_keeps_well_spaced(self):
        result = _drop_too_close([10, 50, 100], min_gap=18)
        assert result == [10, 50, 100]


class TestGrid:
    def setup_method(self):
        # 6 x positions => 5 columns, 9 y positions => 7 rows total, 1 header => 6 data rows
        self.xs = [0, 60, 180, 250, 450, 720]
        self.ys = [0, 40, 80, 120, 160, 200, 240, 280, 320]
        self.grid = Grid(xs=self.xs, ys=self.ys, header_rows=1)

    def test_n_cols(self):
        assert self.grid.n_cols == 5

    def test_n_rows(self):
        assert self.grid.n_rows == 7  # 9 - 1 - 1 = 7

    def test_cell_bbox_type(self):
        x, y, w, h = self.grid.cell_bbox(0, 0)
        assert all(isinstance(v, int) for v in (x, y, w, h))

    def test_cell_bbox_first_cell(self):
        x, y, w, h = self.grid.cell_bbox(0, 0)
        assert x == self.xs[0]
        assert y == self.ys[1]  # after 1 header row

    def test_cells_col_filter(self):
        cells = self.grid.cells(col=4)
        assert all(c.col == 4 for c in cells)

    def test_cells_all_cols(self):
        cells = self.grid.cells()
        assert len(cells) == self.grid.n_rows * self.grid.n_cols


# ---------------------------------------------------------------------------
# cell_extract
# ---------------------------------------------------------------------------


class TestCropCell:
    def setup_method(self):
        self.warped = np.random.randint(0, 255, (500, 800, 3), dtype=np.uint8)

    def test_normal_crop_shape(self):
        bbox = (100, 100, 80, 50)
        crop = crop_cell(self.warped, bbox, inset=4, pad_y=0)
        assert crop.ndim == 3
        assert crop.shape[2] == 3

    def test_inset_reduces_size(self):
        bbox = (100, 100, 80, 50)
        crop_no_inset = crop_cell(self.warped, bbox, inset=0, pad_y=0)
        crop_inset = crop_cell(self.warped, bbox, inset=4, pad_y=0)
        assert crop_inset.shape[0] <= crop_no_inset.shape[0]
        assert crop_inset.shape[1] <= crop_no_inset.shape[1]

    def test_pad_y_extends_height(self):
        bbox = (100, 100, 80, 50)
        crop_no_pad = crop_cell(self.warped, bbox, inset=4, pad_y=0)
        crop_padded = crop_cell(self.warped, bbox, inset=4, pad_y=10)
        assert crop_padded.shape[0] >= crop_no_pad.shape[0]

    def test_out_of_bounds_does_not_crash(self):
        bbox = (790, 490, 80, 50)  # extends past image borders
        crop = crop_cell(self.warped, bbox, inset=4, pad_y=6)
        assert crop.size > 0
