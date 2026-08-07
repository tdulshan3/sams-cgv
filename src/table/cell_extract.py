"""Cell extraction pipeline stage.

Wires line_detect and grid_builder into the Stage interface. Crops the
signature column from the warped colour image and writes the results to
ctx["grid"] and ctx["cells"].

Cell crops come from ctx["warped"] (colour), never from ctx["binary"].
M6 needs the pen colour to do ink segmentation.
"""

from __future__ import annotations

import cv2
import numpy as np

from src import config
from src.models import Cell
from src.table.grid_builder import Grid, _group_into_bands, build_grid, select_student_table
from src.table.line_detect import (
    detect_horizontal_lines,
    detect_lines_hough,
    detect_vertical_lines,
    line_mask,
)
from src.utils.logging import get_logger
from src.utils.stage import Stage

log = get_logger("table")


def crop_cell(
    warped: np.ndarray,
    bbox: tuple[int, int, int, int],
    inset: int = config.CELL_INSET,
    pad_y: int = 0,
) -> np.ndarray:
    """Crop one cell from the warped colour image.

    Args:
        warped: Full colour warped sheet (BGR uint8).
        bbox: (x, y, w, h) of the cell in the warped image.
        inset: Pixels to shave off each side so the border line is excluded.
        pad_y: Extra pixels to add below the cell for overflowing signatures.

    Returns:
        BGR uint8 crop. Returns a 1x1 black image if the clipped region is empty.
    """
    x, y, w, h = bbox
    h_img, w_img = warped.shape[:2]

    x1 = min(max(x + inset, 0), w_img - 1)
    x2 = min(max(x + w - inset, x1 + 1), w_img)
    y1 = min(max(y + inset, 0), h_img - 1)
    y2 = min(max(y + h - inset + pad_y, y1 + 1), h_img)

    crop = warped[y1:y2, x1:x2]
    if crop.size == 0:
        log.warning("empty crop for bbox %s after inset/pad; returning 1x1 placeholder", bbox)
        return np.zeros((1, 1, 3), dtype=np.uint8)
    return crop


def _draw_grid_overlay(warped: np.ndarray, grid: Grid) -> np.ndarray:
    """Draw detected grid lines over the warped sheet for the report figure."""
    overlay = warped.copy()
    for x in grid.xs:
        cv2.line(overlay, (x, 0), (x, overlay.shape[0]), (0, 255, 0), 2)
    for y in grid.ys:
        cv2.line(overlay, (0, y), (overlay.shape[1], y), (0, 0, 255), 2)
    return overlay


def _draw_two_tables(warped: np.ndarray, bands: list[list[int]]) -> np.ndarray:
    """Highlight each table band in a different colour for the report figure."""
    colours = [(0, 165, 255), (0, 255, 0), (255, 0, 0)]
    overlay = warped.copy()
    for i, band in enumerate(bands):
        colour = colours[i % len(colours)]
        for y in band:
            cv2.line(overlay, (0, y), (overlay.shape[1], y), colour, 3)
    return overlay


def _clip_to_table_width(
    xs: list[int], band_h_mask: np.ndarray, tolerance: int = 12
) -> list[int]:
    """Drop vertical lines that fall outside the table's own width.

    Vertical detection inside the student band still picks up the edges of the
    sheet and of the photograph, which run the full height of everything and so
    survive any morphology. They arrive as extra entries at both ends of ``xs``,
    and because the signature column is addressed by index, two spurious lines
    on the left silently shift every column: the crop that should hold a
    signature comes back holding the student's name.

    The row rules are the fix. They span exactly the width of the table and
    nothing else, so any column rule must lie between their endpoints.

    Args:
        xs: Detected vertical line positions.
        band_h_mask: Horizontal line mask, cropped to the student table's rows.
        tolerance: Slack in pixels at each end.

    Returns:
        Only the vertical lines that lie within the row rules' span.
    """
    ink_per_column = band_h_mask.sum(axis=0)
    occupied = np.flatnonzero(ink_per_column > 0)
    if occupied.size == 0:
        return xs

    left = int(occupied[0]) - tolerance
    right = int(occupied[-1]) + tolerance
    kept = [x for x in xs if left <= x <= right]
    dropped = len(xs) - len(kept)
    if dropped:
        log.info(
            "dropped %d vertical line(s) outside the table span %d-%d",
            dropped,
            left,
            right,
        )
    return kept


class TableStage(Stage):
    """Pipeline stage: detect the student table and crop signature cells."""

    name = "table"

    def __init__(self) -> None:
        self._h_mask: np.ndarray | None = None
        self._v_mask: np.ndarray | None = None
        self._grid_overlay: np.ndarray | None = None
        self._two_tables: np.ndarray | None = None
        self._cells_preview: np.ndarray | None = None

    def run(self, ctx: dict) -> dict:
        binary: np.ndarray = ctx["binary"]
        warped: np.ndarray = ctx["warped"]

        # --- Horizontal lines and the two table bands, over the whole page ---
        h_mask = line_mask(binary, "horizontal")
        self._h_mask = h_mask
        all_ys = detect_horizontal_lines(binary)

        bands = _group_into_bands(all_ys, gap_threshold=40)
        self._two_tables = _draw_two_tables(warped, bands)
        ys = select_student_table(bands)

        # --- Vertical lines, inside the student table only ---
        #
        # Searching the whole page finds the lecture header table's columns and
        # the page edges as well, and mixes all of them into one x list. It also
        # makes V_KERNEL_RATIO meaningless: a column rule spans the height of
        # its own table, roughly a seventh of the page, so a kernel measured
        # against the full page height erodes every one of them away.
        # Restricted to the band, the ratio is a fraction of the table's own
        # height, which is what it was always meant to be.
        top, bottom = (ys[0], ys[-1]) if len(ys) >= 2 else (0, binary.shape[0])
        strip = binary[top : bottom + 1, :]
        v_mask = line_mask(strip, "vertical")
        self._v_mask = v_mask
        xs = detect_vertical_lines(strip)
        xs = _clip_to_table_width(xs, h_mask[top : bottom + 1, :])

        # --- Hough cross-check (logged but not used for the main grid) ---
        hough_ys, hough_xs = detect_lines_hough(binary)
        log.info(
            "morphology: %d h-lines, %d v-lines | hough: %d h-lines, %d v-lines",
            len(all_ys),
            len(xs),
            len(hough_ys),
            len(hough_xs),
        )

        # --- Build the grid with repair ---
        grid = build_grid(xs, ys)
        self._grid_overlay = _draw_grid_overlay(warped, grid)
        ctx["grid"] = grid

        # --- Crop signature cells (colour, from warped) ---
        cells: list[Cell] = []
        for row in range(grid.n_rows):
            bbox = grid.cell_bbox(row, config.SIGNATURE_COL)
            crop = crop_cell(warped, bbox, inset=config.CELL_INSET, pad_y=config.CELL_PAD_Y)
            cell = Cell(row=row, col=config.SIGNATURE_COL, bbox=bbox, image=crop)
            cells.append(cell)

        log.info("produced %d signature cells", len(cells))
        if len(cells) != config.EXPECTED_DATA_ROWS:
            log.warning(
                "expected %d signature cells but produced %d",
                config.EXPECTED_DATA_ROWS,
                len(cells),
            )

        ctx["cells"] = cells

        # --- Build a preview of all cropped cells side by side ---
        self._cells_preview = _make_cells_preview(cells)

        return ctx

    def figures(self) -> dict[str, np.ndarray]:
        figs: dict[str, np.ndarray] = {}
        if self._h_mask is not None:
            figs["h_line_mask"] = self._h_mask
        if self._v_mask is not None:
            figs["v_line_mask"] = self._v_mask
        if self._two_tables is not None:
            figs["two_tables"] = self._two_tables
        if self._grid_overlay is not None:
            figs["grid_overlay"] = self._grid_overlay
        if self._cells_preview is not None:
            figs["cells"] = self._cells_preview
        return figs


def _make_cells_preview(cells: list[Cell]) -> np.ndarray | None:
    """Stack cell crops horizontally for the progress viewer."""
    crops = [c.image for c in cells if c.image is not None]
    if not crops:
        return None
    target_h = max(c.shape[0] for c in crops)
    padded: list[np.ndarray] = []
    for crop in crops:
        h, w = crop.shape[:2]
        if h < target_h:
            pad = np.zeros((target_h - h, w, 3), dtype=np.uint8)
            crop = np.vstack([crop, pad])
        padded.append(crop)
    return np.hstack(padded)
