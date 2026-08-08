"""The verdict: was this box signed, and who does it belong to?

Last stage of the pipeline. It looks at no pixels of its own — M6 has already
measured every signature cell, and this module only reads those numbers, names
the student each cell belongs to, and writes the result to the database.

**Why the rule is not one number.** Ink coverage alone gets two of our thirty
cells wrong, and both are known in advance (BUILD_SPEC.md §4 deviation 7): on
``21.06.2019`` the last cell holds the lecturer's handwritten ``ab``, and on
``05.07.2019`` one cell holds a stray red pen tick. Both are ink. Both mean
*absent*. So the rule asks three things of a cell — enough coverage, at least
one real connected component, and enough skeleton length that the pen actually
travelled somewhere — and reports how sure it is rather than pretending a cell
sitting on the threshold is as clear-cut as an empty one.
"""

from __future__ import annotations

import numpy as np

from src import config
from src.models import Cell, InkResult, Student
from src.utils.logging import get_logger

log = get_logger("decision")


def decide(result: InkResult) -> tuple[bool, float]:
    """Judge one signature cell.

    All four conditions must hold for a cell to count as signed::

        ink_ratio     >= INK_RATIO_THRESHOLD
        ink_ratio     <= MAX_INK_RATIO
        components    >= MIN_COMPONENTS
        stroke_length >= MIN_STROKE_LENGTH

    Every one of those numbers comes from the sweep in
    ``tools/tune_threshold.py`` against the 30 labelled cells, and that tool
    re-implements this rule so the two can be checked against each other —
    ``tests/test_decision.py::test_sweep_rule_matches_decide`` fails if they
    ever drift apart.

    Args:
        result: What M6 measured inside the cell.

    Returns:
        ``(present, confidence)``. Confidence runs from 0.5 to 1.0: 1.0 when
        the ink ratio is clear of :data:`~src.config.CONF_LOW` or
        :data:`~src.config.CONF_HIGH`, falling towards 0.5 as the cell
        approaches the threshold from either side.

        A cell that clears the ink threshold but fails on shape — the ``ab``,
        the stray tick — is called absent with confidence 0.5. That is
        deliberate. The verdict is the right one, the evidence for it is thin,
        and saying so puts the cell in front of a human instead of burying it.
    """
    ink_ratio = float(result.ink_ratio)
    has_ink = ink_ratio >= config.INK_RATIO_THRESHOLD
    looks_written = (
        ink_ratio <= config.MAX_INK_RATIO
        and result.components >= config.MIN_COMPONENTS
        and result.stroke_length >= config.MIN_STROKE_LENGTH
    )
    present = has_ink and looks_written

    if present:
        span = config.CONF_HIGH - config.INK_RATIO_THRESHOLD
        distance = (ink_ratio - config.INK_RATIO_THRESHOLD) / span if span > 0 else 1.0
    elif has_ink:
        # Enough ink, but the shape says it is not a signature. The two
        # conditions disagree, so this is the least certain verdict we make.
        distance = 0.0
    else:
        span = config.INK_RATIO_THRESHOLD - config.CONF_LOW
        distance = (config.INK_RATIO_THRESHOLD - ink_ratio) / span if span > 0 else 1.0

    confidence = 0.5 + 0.5 * float(np.clip(distance, 0.0, 1.0))
    return present, round(confidence, 3)


def map_rows_to_students(cells: list[Cell], students: list[Student]) -> dict[int, str]:
    """Work out which student each sheet row belongs to.

    Positional. The XML was transcribed off the sheets in row order (T0), so
    row *n* is student *n*, and OCR of the printed index column would add a
    failure mode without adding information.

    What it does *not* do is assume the two lists are the same length. If M5
    found five rows where the roll has six, the sixth student simply gets no
    record and the run warns rather than quietly shifting every student up a
    row — which would be the same kind of wrong as reading the lecturer's
    signature as a student's.

    Args:
        cells: The signature-column cells, in row order.
        students: The roll, in sheet row order.

    Returns:
        ``{row: student_index}`` for every row that has a student.
    """
    rows = sorted({cell.row for cell in cells})
    if len(rows) != len(students):
        log.warning(
            "%d rows detected but %d students in info.xml — "
            "mapping the first %d by position, the rest have no record",
            len(rows),
            len(students),
            min(len(rows), len(students)),
        )

    mapping: dict[int, str] = {}
    for row, student in zip(rows, students):
        mapping[row] = student.index
        student.row = row
    return mapping
