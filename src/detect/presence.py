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

from pathlib import Path

import numpy as np

from src import config
from src.io.db import Database
from src.models import AttendanceRecord, Cell, InkResult, SheetMeta, Student
from src.utils.logging import get_logger
from src.utils.stage import Stage

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


def _rename_to_index(path_text: str | None, student_index: str) -> str | None:
    """Rename one of M6's ``row_<n>`` crops to ``<index>``, and say where it is.

    M6 saves each crop as ``outputs/cells/<date>/row_<n>.png`` because at that
    point in the pipeline nothing knows the student's index — the mapping is
    this module's job. But ``investigate.py`` finds a student's samples by
    globbing ``outputs/cells/<date>/<index>.png`` (spec §7, and the hand-off
    table in §14), so somebody has to bridge the two, and the first place with
    both facts in hand is here.

    Args:
        path_text: Where M6 wrote the file, or ``None``.
        student_index: Who the row turned out to belong to.

    Returns:
        The path the file now lives at, or ``None`` if there was no file.
    """
    if not path_text:
        return None
    source = Path(path_text)
    if not source.is_file():
        return path_text
    suffix = "_mask" if source.stem.endswith("_mask") else ""
    target = source.with_name(f"{student_index}{suffix}{source.suffix}")
    if target != source:
        source.replace(target)
    return str(target)


class DecisionStage(Stage):
    """M7 — decide present or absent, then persist the whole sheet."""

    name = "decision"

    def __init__(self, db: Database | None = None) -> None:
        """
        Args:
            db: Where to write. Tests pass a temporary database; the pipeline
                passes nothing and gets ``data/attendance.db``.
        """
        self.db = db or Database()
        self._figures: dict[str, np.ndarray] = {}

    def run(self, ctx: dict) -> dict:
        """Turn M6's measurements into attendance records, and store them."""
        ink: list[InkResult] = ctx.get("ink") or []
        students: list[Student] = ctx.get("students") or []
        sheet: SheetMeta | None = ctx.get("sheet")

        if not ink:
            log.warning("no ink results to decide on — no attendance recorded")
            ctx["records"] = []
            return ctx
        if not students:
            log.warning("no students read from info.xml — no attendance recorded")
            ctx["records"] = []
            return ctx

        sheet_date = sheet.date if sheet else "unknown"
        mapping = map_rows_to_students([result.cell for result in ink], students)

        records: list[AttendanceRecord] = []
        signatures: list[tuple[str, dict]] = []
        for result in ink:
            student_index = mapping.get(result.cell.row)
            if student_index is None:
                continue
            result.cell.student_index = student_index

            present, confidence = decide(result)
            records.append(
                AttendanceRecord(
                    student_index=student_index,
                    sheet_date=sheet_date,
                    present=present,
                    confidence=confidence,
                    ink_ratio=float(result.ink_ratio),
                )
            )
            log.debug(
                "row %d -> %s: ink=%.4f components=%d stroke=%d -> %s (%.2f)",
                result.cell.row,
                student_index,
                result.ink_ratio,
                result.components,
                result.stroke_length,
                "present" if present else "absent",
                confidence,
            )

            result.crop_path = _rename_to_index(result.crop_path, student_index)
            result.mask_path = _rename_to_index(result.mask_path, student_index)
            signatures.append(
                (
                    student_index,
                    {
                        "crop_path": result.crop_path,
                        "mask_path": result.mask_path,
                        "ink_ratio": float(result.ink_ratio),
                        "components": int(result.components),
                        "aspect": float(result.aspect),
                        "stroke_length": int(result.stroke_length),
                    },
                )
            )

        ctx["records"] = records
        if sheet is not None:
            self._persist(sheet, students, records, signatures, ctx)
        self._figures = {"decision": _verdict_strip(ink, records)}

        present = sum(1 for record in records if record.present)
        log.info(
            "%d records: %d present, %d absent (%d uncertain)",
            len(records),
            present,
            len(records) - present,
            sum(1 for record in records if record.confidence < config.UNCERTAIN_BELOW),
        )
        return ctx

    def _persist(
        self,
        sheet: SheetMeta,
        students: list[Student],
        records: list[AttendanceRecord],
        signatures: list[tuple[str, dict]],
        ctx: dict,
    ) -> None:
        """Write students, the sheet, every verdict and every signature crop.

        Students go in before attendance because attendance holds a foreign key
        to them, and a signature row without an attendance row would be a
        signature belonging to nobody.
        """
        subject_code = ctx.get("subject_code") or sheet.subject_code
        self.db.init_schema()
        self.db.upsert_students(students)
        sheet_id = self.db.upsert_sheet(sheet, subject_code=subject_code)
        self.db.save_attendance(records, sheet_id)
        for student_index, fields in signatures:
            self.db.save_signature(student_index, sheet_id, **fields)
        log.info("sheet %s stored as id %d in %s", sheet.date, sheet_id, self.db.path)

    def figures(self) -> dict[str, np.ndarray]:
        """One strip of every signature cell, framed green for present, red for absent."""
        return {name: image for name, image in self._figures.items() if image is not None}


def _verdict_strip(ink: list[InkResult], records: list[AttendanceRecord]) -> np.ndarray | None:
    """Stack the signature cells into one image, each framed by its verdict.

    Cheap to build and the fastest way to check a run by eye: six crops, six
    coloured frames, and any disagreement with the sheet is obvious at a
    glance.
    """
    verdicts = {record.student_index: record.present for record in records}
    tiles: list[np.ndarray] = []
    width = max(
        (result.cell.image.shape[1] for result in ink if result.cell.image is not None),
        default=0,
    )
    if width == 0:
        return None

    for result in ink:
        image = result.cell.image
        if image is None or image.size == 0:
            continue
        tile = np.zeros((image.shape[0], width, 3), dtype=np.uint8)
        tile[:, : image.shape[1]] = image[:, :width]
        present = verdicts.get(result.cell.student_index or "", False)
        colour = (0, 170, 0) if present else (0, 0, 200)  # BGR
        tile[:3, :] = colour
        tile[-3:, :] = colour
        tile[:, :3] = colour
        tile[:, -3:] = colour
        tiles.append(tile)

    return np.vstack(tiles) if tiles else None
