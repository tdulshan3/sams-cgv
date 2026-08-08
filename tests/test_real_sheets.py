"""End-to-end checks against the five real signing sheets.

Every other test file in this project is synthetic, and synthetic tests have
now let three separate defects reach ``main``:

* M2 shipped a geometry stage that mirrored every sheet and cropped three of
  them to slivers of under 1% of the page. 5 tests passed.
* M5 shipped a table stage that crashed on the first real photo and, once the
  crash was fixed, produced zero cells on all five sheets. 84 tests passed.
* Both shipped ``cv2.HoughLinesP`` unpacking that only fails on an image with
  enough lines in it to return a result.

Each was found by running ``sams.py`` by hand. These tests are that run, made
automatic. They are deliberately about *outcomes* — how many cells, of what
shape, in which column — rather than about any module's internals, so they stay
valid as M2 and M5 keep tuning.

They are slower than the rest of the suite (a few seconds per sheet). That is
the correct trade: the suite is worth nothing if it cannot catch the bug class
that has actually bitten us.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src import config  # noqa: E402
from src.models import SheetMeta  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402

SHEETS = sorted(config.SHEETS.glob("*.png"))

pytestmark = pytest.mark.skipif(not SHEETS, reason="data/sheets is empty")


def _run(sheet: Path) -> dict:
    """Run the real pipeline exactly as ``sams.py`` assembles it."""
    import sams

    stages = [make_stage() for make_stage in sams.STAGES]
    students = sams.load_students(config.INFO_XML)
    return Pipeline(stages).run(SheetMeta(path=sheet, date=sheet.stem), students)


@pytest.fixture(scope="module", params=SHEETS, ids=lambda p: p.stem)
def processed(request) -> dict:
    return _run(request.param)


def test_pipeline_completes_on_every_sheet(processed: dict) -> None:
    """No stage raises, and every contract key is present."""
    for key in ("bgr", "warped", "grey", "binary", "grid", "cells"):
        assert key in processed, f"stage output {key!r} missing from ctx"


def test_geometry_keeps_the_whole_page(processed: dict) -> None:
    """The corrected sheet is not a sliver.

    M2's regression cropped to the student table, or to a fragment of it, at
    0.4% of the photo. Anything under half the original area means the outline
    detection has locked onto something inside the page instead of the page.
    """
    bgr, warped = processed["bgr"], processed["warped"]
    original_area = bgr.shape[0] * bgr.shape[1]
    warped_area = warped.shape[0] * warped.shape[1]

    assert warped_area > 0.5 * original_area, (
        f"warped is {warped_area / original_area:.1%} of the photo — "
        "geometry has cropped to something smaller than the sheet"
    )


def test_six_signature_cells_per_sheet(processed: dict) -> None:
    """The acceptance test from BUILD_SPEC.md section 9.5."""
    cells = processed["cells"]
    assert len(cells) == config.EXPECTED_DATA_ROWS
    assert [cell.row for cell in cells] == list(range(config.EXPECTED_DATA_ROWS))


def test_grid_has_the_measured_shape(processed: dict) -> None:
    """Five columns and six data rows, as measured in T0."""
    grid = processed["grid"]
    assert grid.n_cols == config.EXPECTED_COLS
    assert grid.n_rows == config.EXPECTED_DATA_ROWS


def test_cells_are_colour_crops_of_the_signature_column(processed: dict) -> None:
    """M6 needs BGR, not binary — pen colour is their whole method.

    The column index matters as much as the count: when spurious vertical lines
    shifted the grid, six cells were still produced and every one of them held a
    student's printed name.
    """
    for cell in processed["cells"]:
        assert cell.col == config.SIGNATURE_COL
        assert cell.image is not None and cell.image.ndim == 3
        assert cell.image.dtype == np.uint8
        assert cell.image.shape[0] > 5 and cell.image.shape[1] > 5


def test_signature_cells_sit_in_the_rightmost_column(processed: dict) -> None:
    """A cross-check on the column index that does not rely on it.

    Signature is the last column on the printed sheet, so its cells must start
    further right than every other column's. This catches an off-by-one in the
    grid even if ``SIGNATURE_COL`` and the detected column count agree.
    """
    grid = processed["grid"]
    signature_left = grid.cell_bbox(0, config.SIGNATURE_COL)[0]
    others = [grid.cell_bbox(0, col)[0] for col in range(grid.n_cols) if col != config.SIGNATURE_COL]

    assert signature_left > max(others)


def test_ink_results_carry_every_feature_m7_needs(processed: dict) -> None:
    """All seven features reach M7, not just the five that fit the old dataclass.

    ``filled_ratio`` and ``centroid_offset`` were being computed by M6 and then
    dropped at the ``InkResult`` boundary. They are the two that separate a
    signature from ink that is not one — on ``21.06.2019`` the lecturer's
    handwritten ``ab`` has a higher ink ratio than any real signature on any
    sheet, so ink ratio alone cannot reject it.
    """
    ink = processed["ink"]
    assert len(ink) == len(processed["cells"])

    signed = [r for r in ink if r.ink_ratio > 0]
    assert signed, "no cell on this sheet produced any ink at all"
    for result in signed:
        assert result.stroke_bbox is not None
        assert result.aspect > 0
        assert result.stroke_length > 0
        assert result.filled_ratio > 0, "filled_ratio is being dropped again"


def test_signature_crops_are_named_by_student_index(processed: dict) -> None:
    """``outputs/cells/<date>/<index>.png`` — BUILD_SPEC.md section 5.4.

    ``investigate.py`` counts a student's samples by globbing this exact name.
    Named by row number instead, every student appears to have zero signatures
    and M8's whole command is dead on arrival.
    """
    from src import cli

    indices = {student.index for student in processed["students"]}
    if not indices:
        pytest.skip("no roll available, so crops cannot be named by index")

    for index in indices:
        assert cli.signature_samples(index), f"no saved crop found for {index}"

    assert set(cli.known_indices()) == indices


def test_binary_keeps_the_ink_is_white_convention(processed: dict) -> None:
    """Section 2: ink is 255, paper is 0, on every sheet."""
    binary = processed["binary"]
    assert set(np.unique(binary)) <= {0, 255}

    ink_fraction = float((binary > 0).mean())
    assert 0.005 < ink_fraction < 0.40, (
        f"{ink_fraction:.1%} of the page is ink — the polarity is probably inverted"
    )
