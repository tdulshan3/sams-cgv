"""Shared data contracts.

Every module in the project passes these objects, and only these objects, to
every other module. They are defined once here so that nine people can work in
parallel without agreeing anything else in person.

Two rules that the whole project depends on:

* A student index is a **string**. ``"007"`` and ``"10000409"``, never ``7``.
  Leading zeros are part of the identifier and integers destroy them.
* An image is a NumPy array in OpenCV's own conventions: ``uint8``, BGR channel
  order for colour, and **ink is 255 (white) on paper 0 (black)** once the
  pipeline has binarised it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Student:
    """One student, as read from ``info.xml``."""

    index: str
    """Student number exactly as printed, e.g. ``"10000409"``. Always a string."""

    name: str
    """Full name as printed on the sheet."""

    row: int | None = None
    """Zero based row this student occupies on the sheet. Filled in by M7."""


@dataclass
class SheetMeta:
    """The signing sheet being processed, and where it came from."""

    path: Path
    """Path to the photo on disk."""

    date: str
    """Sheet date taken from the filename stem, e.g. ``"12.07.2019"``."""

    subject_code: str = ""
    """Subject code from ``info.xml``, e.g. ``"CS402.3"``. Empty until read."""


@dataclass
class Cell:
    """One box of the printed student table, cut out of the flattened sheet."""

    row: int
    """Zero based data row. The header row is not a data row and is never 0."""

    col: int
    """Column: ``0=No, 1=Student No, 2=Title, 3=Student Name, 4=Signature``.

    Measured in task T0 — the real sheet carries a ``Title`` column that the
    first draft of the spec did not know about, which is why the signature
    column is 4 and not 3. Use :data:`src.config.SIGNATURE_COL`, never a
    literal.
    """

    bbox: tuple[int, int, int, int]
    """``(x, y, w, h)`` in the warped image, in pixels."""

    image: np.ndarray | None = None
    """BGR crop of this cell, taken from ``ctx["warped"]``."""

    student_index: str | None = None
    """Which student this cell belongs to. Attached by M7, not by M5."""


@dataclass
class InkResult:
    """What M6 found inside one signature cell.

    The decision stage reads these numbers; it never looks at pixels itself.
    """

    cell: Cell
    """The cell these measurements describe."""

    mask: np.ndarray | None = None
    """``uint8`` mask of the same size as the cell crop, ink = 255."""

    ink_ratio: float = 0.0
    """Ink pixels divided by cell pixels, in ``[0, 1]``."""

    components: int = 0
    """Connected components in the mask. A speck is one, a signature is a few."""

    stroke_bbox: tuple[int, int, int, int] | None = None
    """``(x, y, w, h)`` bounding box of all ink, relative to the cell."""

    aspect: float = 0.0
    """Width over height of :attr:`stroke_bbox`. Signatures are wide, ticks are not."""

    stroke_length: int = 0
    """Pixels in the skeletonised stroke — how much pen travelled, not how thick."""

    filled_ratio: float = 0.0
    """Ink pixels divided by :attr:`stroke_bbox` area, in ``[0, 1]``.

    How solidly the ink fills its own bounding box. A signature is sparse — a
    thin line wandering through a wide box — while a written word, a smudge or
    a tick is dense. On ``21.06.2019`` the lecturer's handwritten ``ab`` has a
    *higher* ink ratio than any real signature on any sheet, so ink ratio alone
    cannot reject it. This is one of the two features that can.
    """

    centroid_offset: float = 0.0
    """Distance from the ink's centre of mass to the cell centre, normalised.

    A signature is written across the box. A stray tick sits wherever the pen
    happened to touch. Together with :attr:`filled_ratio` this is what M7 needs
    to separate ink that means *present* from ink that does not.
    """

    crop_path: str | None = None
    """Where the BGR crop was written, under ``outputs/cells/``."""

    mask_path: str | None = None
    """Where the ink mask was written, under ``outputs/cells/``."""


@dataclass
class AttendanceRecord:
    """The pipeline's verdict for one student on one sheet."""

    student_index: str
    """Matches :attr:`Student.index`."""

    sheet_date: str
    """Matches :attr:`SheetMeta.date`."""

    present: bool
    """``True`` when the signature cell was judged to hold a signature."""

    confidence: float
    """How sure the decision stage is, in ``[0, 1]``.

    Values near 0.5 are the uncertain ones and are counted separately in the
    summary table so a human can check them.
    """

    ink_ratio: float = 0.0
    """Carried through from :attr:`InkResult.ink_ratio` for the report charts."""
