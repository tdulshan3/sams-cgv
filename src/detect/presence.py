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

from src.models import Cell, Student
from src.utils.logging import get_logger

log = get_logger("decision")


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
