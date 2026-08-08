#!/usr/bin/env python3
"""SAMS — read a signing sheet photo and work out who was present.

The first of the three commands the coursework brief fixes::

    python sams.py data/sheets/12.07.2019.png data/info.xml

It shows each image processing step as it happens, saves those steps as
numbered images for the report, and writes the attendance to the local
database.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src import config
from src.detect.ink_mask import InkStage
from src.detect.presence import DecisionStage
from src.models import AttendanceRecord, SheetMeta, Student
from src.pipeline import Pipeline
from src.preprocess.binarize import BinarizeStage
from src.preprocess.deskew import GeometryStage
from src.preprocess.enhance import EnhanceStage
from src.table.cell_extract import TableStage
from src.utils.logging import collect_warnings, get_logger, set_debug
from src.utils.stage import Stage
from src.viz.progress import ProgressViewer

log = get_logger("sams")

SUMMARY_RULE = "─" * 61
"""Width of the summary box. Fits an 80 column terminal with room to spare."""

MAX_SUMMARY_WARNINGS = 5
"""Warnings listed in full before the rest are counted instead."""

STAGES: list[Callable[[], Stage]] = [
    GeometryStage,        # M2 — real, merged in #8
    EnhanceStage,         # M3 — real, merged in #9
    BinarizeStage,        # M4 — real, merged in #10
    TableStage,           # M5 — real, merged in #11
    InkStage,             # M6 — real, src.detect.ink_mask
    DecisionStage,        # M7 — real, src.detect.presence
]

"""The pipeline, in the fixed order from BUILD_SPEC.md section 6.3.

Swapping a stub for the real module is one line here and one deletion in
``src/stubs.py``. Nothing else in the project changes, which is the whole point
of every stage being a :class:`~src.utils.stage.Stage`.
"""


@dataclass
class RunSummary:
    """What one run of the pipeline produced, in the form the user reads.

    Built from the finished context rather than accumulated as the run goes,
    so the numbers cannot drift away from the records that were actually
    written.
    """

    sheet_date: str
    students: int
    records: list[AttendanceRecord]
    duration: float
    steps_dir: Path
    step_count: int
    warnings: list[str] = field(default_factory=list)

    @property
    def present(self) -> int:
        """Records judged to hold a signature."""
        return sum(1 for record in self.records if record.present)

    @property
    def absent(self) -> int:
        """Records judged to have an empty signature cell."""
        return len(self.records) - self.present

    @property
    def uncertain(self) -> int:
        """Records a human should check, by :data:`src.config.UNCERTAIN_BELOW`.

        These are counted as present or absent as well — the number is a
        prompt to look, not a third verdict.
        """
        return sum(
            1
            for record in self.records
            if record.confidence < config.UNCERTAIN_BELOW
        )

    def render(self) -> str:
        """The summary block, exactly as it is printed."""
        try:
            steps = self.steps_dir.relative_to(config.ROOT)
        except ValueError:
            steps = self.steps_dir

        lines = [
            SUMMARY_RULE,
            f" Sheet     : {self.sheet_date}",
            f" Students  : {self.students}",
            f" Present   : {self.present}",
            f" Absent    : {self.absent}",
            f" Uncertain : {self.uncertain}",
            f" Duration  : {self.duration:.2f} s",
            f" Steps     : {steps}/  ({self.step_count} images)",
        ]
        if self.warnings:
            lines.append(f" Warnings  : {len(self.warnings)}")
            for message in self.warnings[:MAX_SUMMARY_WARNINGS]:
                lines.append(f"   - {message}")
            remaining = len(self.warnings) - MAX_SUMMARY_WARNINGS
            if remaining > 0:
                lines.append(f"   … and {remaining} more, see the log above")
        lines.append(SUMMARY_RULE)
        return "\n".join(lines)


def load_students(xml_path: Path) -> list[Student]:
    """Read the roll from ``info.xml``."""
    from src.io.xml_parser import parse_students

    students = parse_students(xml_path)
    log.info("%d students read from %s", len(students), xml_path.name)
    return students


def build_parser() -> argparse.ArgumentParser:
    """Command line interface for ``sams.py``."""
    parser = argparse.ArgumentParser(
        prog="sams.py",
        description=(
            "Process a signing sheet photo into attendance records, showing "
            "every image processing step on the way."
        ),
    )
    parser.add_argument("image", type=Path, help="path to a signing sheet photo")
    parser.add_argument("xml", type=Path, help="path to info.xml")
    parser.add_argument(
        "--no-show", action="store_true", help="do not open the montage window"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="do not write step images"
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    return parser


def _available_sheets(limit: int = 10) -> list[str]:
    """Sheet filenames the user could have meant, for a helpful error."""
    if not config.SHEETS.is_dir():
        return []
    return sorted(path.name for path in config.SHEETS.iterdir() if path.is_file())[:limit]


def validate_paths(image: Path, xml: Path) -> None:
    """Stop with a clear message if either input is not a readable file.

    A mistyped filename is a user mistake, not a bug, so it gets one line of
    plain English and a hint — never a traceback. Section 10 of the spec.

    Raises:
        SystemExit: With code 2, the conventional exit code for a usage error.
    """
    if not image.is_file():
        print(f"error: no signing sheet image at {image}")
        sheets = _available_sheets()
        if sheets:
            print("       sheets available in data/sheets:")
            for name in sheets:
                print(f"         {name}")
        else:
            print(f"       no sheets found in {config.SHEETS} either")
        raise SystemExit(2)

    if not xml.is_file():
        print(f"error: no student list at {xml}")
        if config.INFO_XML.is_file():
            try:
                hint = config.INFO_XML.relative_to(Path.cwd())
            except ValueError:
                hint = config.INFO_XML
            print(f"       did you mean {hint} ?")
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    set_debug(args.debug)
    warnings = collect_warnings()
    config.ensure_dirs()
    validate_paths(args.image, args.xml)

    sheet_date = args.image.stem
    log.info("sheet date %s from filename %s", sheet_date, args.image.name)

    sheet = SheetMeta(path=args.image, date=sheet_date)
    students = load_students(args.xml)

    viewer = ProgressViewer(
        sheet_date=sheet_date,
        show=config.SHOW_PROGRESS and not args.no_show,
        save=config.SAVE_STEPS and not args.no_save,
    )
    pipeline = Pipeline([make_stage() for make_stage in STAGES], viewer=viewer)
    ctx = pipeline.run(sheet, students)

    written = viewer.save_all()

    summary = RunSummary(
        sheet_date=sheet_date,
        students=len(students),
        records=list(ctx.get("records") or []),
        duration=pipeline.duration(),
        steps_dir=viewer.output_dir,
        step_count=len(written),
        warnings=warnings.messages,
    )
    print()
    print(summary.render())

    viewer.show_montage()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
