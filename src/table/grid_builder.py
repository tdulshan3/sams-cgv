"""Grid dataclass and builder.

Takes detected line positions and turns them into a Grid that knows how
to produce Cell bounding boxes. The key decision here is selecting the
student table from the two tables on the page.

Sheet layout:
  - One-row lecture header table near the top (Date / Time / Lecturer / Signature)
  - Student table below (7 horizontal lines = header + 6 data rows)

The lecture header table is discarded. If the student table is taken, the
lecturer's own signature would be reported as a student's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import (
    EXPECTED_COLS,
    EXPECTED_DATA_ROWS,
    MIN_COL_WIDTH,
    MIN_ROW_HEIGHT,
)
from src.models import Cell
from src.utils.logging import get_logger

log = get_logger("table")


@dataclass
class Grid:
    """Detected grid: vertical line positions and horizontal line positions.

    ``xs`` and ``ys`` are the outer borders included. A grid with 6 vertical
    lines and 8 horizontal lines gives 5 columns and 7 rows.
    """

    xs: list[int]
    ys: list[int]
    header_rows: int = 1

    @property
    def n_cols(self) -> int:
        """Number of data columns."""
        return len(self.xs) - 1

    @property
    def n_rows(self) -> int:
        """Number of data rows, header excluded."""
        return len(self.ys) - 1 - self.header_rows

    def cell_bbox(self, row: int, col: int) -> tuple[int, int, int, int]:
        """Bounding box (x, y, w, h) for one data cell.

        ``row`` is zero-based after the header. ``col`` is zero-based.
        """
        y_idx = row + self.header_rows
        x = self.xs[col]
        y = self.ys[y_idx]
        w = self.xs[col + 1] - x
        h = self.ys[y_idx + 1] - y
        return x, y, w, h

    def cells(self, col: int | None = None) -> list[Cell]:
        """Build Cell objects for every data row.

        Args:
            col: Column index to extract. If ``None``, all columns are returned.
        """
        cols = range(self.n_cols) if col is None else [col]
        result: list[Cell] = []
        for r in range(self.n_rows):
            for c in cols:
                bbox = self.cell_bbox(r, c)
                result.append(Cell(row=r, col=c, bbox=bbox))
        return result


def _group_into_bands(ys: list[int], gap_threshold: int = 40) -> list[list[int]]:
    """Split a flat list of Y positions into bands separated by large gaps.

    Each band becomes one candidate table.
    """
    if not ys:
        return []
    bands: list[list[int]] = [[ys[0]]]
    for y in ys[1:]:
        if y - bands[-1][-1] > gap_threshold:
            bands.append([])
        bands[-1].append(y)
    return bands


def select_student_table(row_bands: list[list[int]]) -> list[int]:
    """Return the Y positions belonging to the student table.

    The student table has the most horizontal lines. The lecture header
    table is a small band (usually 2 lines) near the top.

    Logs which band was chosen and warns when the expected two bands are
    not found.
    """
    if not row_bands:
        log.warning("no horizontal line bands found; cannot select student table")
        return []

    if len(row_bands) == 1:
        log.warning("only one table band found; expected two (header + student)")

    # The student table has the most lines.
    student_band = max(row_bands, key=len)
    log.info(
        "selected student table band: %d lines at y=%s",
        len(student_band),
        student_band,
    )
    other_bands = [b for b in row_bands if b is not student_band]
    for band in other_bands:
        log.info("discarded non-student band: %d lines at y=%s", len(band), band)

    return student_band


def _repair_missing_lines(ys: list[int], min_height: int) -> list[int]:
    """Insert lines where a gap is close to a multiple of the median spacing.

    Real phone photos lose faint printed lines. If a gap is approximately
    2x or 3x the median row height, intermediate lines are inserted.
    """
    if len(ys) < 2:
        return list(ys)

    spacings = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    median_spacing = int(sorted(spacings)[len(spacings) // 2])
    if median_spacing < min_height:
        return list(ys)

    repaired = [ys[0]]
    for i in range(len(ys) - 1):
        gap = ys[i + 1] - ys[i]
        multiple = round(gap / median_spacing)
        if multiple >= 2:
            log.warning(
                "gap of %d px at y=%d is ~%dx median (%d); inserting %d line(s)",
                gap,
                ys[i],
                multiple,
                median_spacing,
                multiple - 1,
            )
            for k in range(1, multiple):
                repaired.append(ys[i] + k * (gap // multiple))
        repaired.append(ys[i + 1])
    return repaired


def _drop_too_close(positions: list[int], min_gap: int) -> list[int]:
    """Remove lines that are closer than ``min_gap`` to their predecessor."""
    if not positions:
        return []
    kept = [positions[0]]
    for p in positions[1:]:
        if p - kept[-1] >= min_gap:
            kept.append(p)
        else:
            log.debug("dropped duplicate line at %d (too close to %d)", p, kept[-1])
    return kept


def build_grid(xs: list[int], ys: list[int]) -> Grid:
    """Build a Grid from raw line positions, repairing obvious gaps.

    Validates the final column and row counts against the expected values
    and warns if they differ.
    """
    ys_clean = _drop_too_close(ys, MIN_ROW_HEIGHT)
    ys_repaired = _repair_missing_lines(ys_clean, MIN_ROW_HEIGHT)
    xs_clean = _drop_too_close(xs, MIN_COL_WIDTH)

    grid = Grid(xs=xs_clean, ys=ys_repaired)

    if grid.n_cols != EXPECTED_COLS:
        log.warning(
            "expected %d columns but found %d; vertical lines: %s",
            EXPECTED_COLS,
            grid.n_cols,
            xs_clean,
        )
    else:
        log.info("column count OK: %d", grid.n_cols)

    if grid.n_rows != EXPECTED_DATA_ROWS:
        log.warning(
            "expected %d data rows but found %d; horizontal lines: %s",
            EXPECTED_DATA_ROWS,
            grid.n_rows,
            ys_repaired,
        )
    else:
        log.info("data row count OK: %d", grid.n_rows)

    return grid
