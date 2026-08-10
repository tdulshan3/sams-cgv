#!/usr/bin/env python3
"""Fill the database with fake attendance so M8 and M9 can start.

The charts and the signature comparison both read from the database, and the
database is not filled until the whole image pipeline runs. Rather than have
two people wait for that, this writes a plausible fake dataset with the real
schema, the real six students and the real five sheet dates:

    python tools/seed_db.py                  # fake rows into data/attendance.db
    python tools/seed_db.py --db /tmp/x.db   # somewhere else
    python tools/seed_db.py --clear          # wipe the fake rows again

Temporary scaffolding, exactly like ``tools/make_fixtures.py``. The numbers are
invented: no image was looked at, so nothing here may end up in the report.
Once ``sams.py`` runs all five sheets for real, run with ``--clear`` and let the
pipeline write the truth.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.io.db import Database  # noqa: E402
from src.models import AttendanceRecord, SheetMeta, Student  # noqa: E402

SEED = 402
"""Fixed so two people seeding separately get the same fake data to talk about."""

FALLBACK_STUDENTS = [
    Student("10000409", "M S Dilshanika Perera"),
    Student("10009301", "C W M A Shehan Abeyrathne"),
    Student("10009302", "B A K M Chithrananda"),
    Student("10009303", "W Shashini Minosha De Silva"),
    Student("10009304", "K L Udara Maduranga Liyanage"),
    Student("10009306", "Hansa Anuradha Wickramanayake"),
]
"""Used when ``info.xml`` cannot be read; the seeder must never be the reason
somebody is blocked."""

SHEET_DATES = ["31.05.2019", "21.06.2019", "28.06.2019", "05.07.2019", "12.07.2019"]

ABSENT_RATE = 0.2
"""Roughly how often the fake student misses a lecture. The real sheets sit
near this, so charts drawn against fake data look like the real ones."""


def load_students() -> list[Student]:
    """The real roll if ``info.xml`` parses, six invented students otherwise."""
    try:
        from src.io.xml_parser import parse_students

        students = parse_students(config.INFO_XML)
        if students:
            return students
    except (ImportError, FileNotFoundError, ValueError) as error:
        print(f"note: falling back to built-in students ({error})")
    return list(FALLBACK_STUDENTS)


def fake_record(student: Student, sheet_date: str, rng: random.Random) -> AttendanceRecord:
    """One invented verdict, with an ink ratio consistent with it."""
    present = rng.random() > ABSENT_RATE
    if present:
        ink_ratio = rng.uniform(0.02, 0.09)
        confidence = rng.uniform(0.75, 0.99)
    else:
        ink_ratio = rng.uniform(0.0, 0.004)
        confidence = rng.uniform(0.7, 0.95)
    return AttendanceRecord(
        student_index=student.index,
        sheet_date=sheet_date,
        present=present,
        confidence=round(confidence, 3),
        ink_ratio=round(ink_ratio, 5),
    )


def seed(db: Database) -> int:
    """Write students, five sheets, and one attendance row per pair.

    Returns:
        How many attendance rows were written.
    """
    rng = random.Random(SEED)
    students = load_students()
    db.init_schema()
    db.upsert_students(students)

    written = 0
    for sheet_date in SHEET_DATES:
        meta = SheetMeta(path=config.SHEETS / f"{sheet_date}.png", date=sheet_date)
        sheet_id = db.upsert_sheet(meta, subject_code="CS402.3")
        records = [fake_record(student, sheet_date, rng) for student in students]
        db.save_attendance(records, sheet_id)
        for record in records:
            if not record.present:
                continue
            folder = config.CELLS / sheet_date
            db.save_signature(
                record.student_index,
                sheet_id,
                crop_path=str(folder / f"{record.student_index}.png"),
                mask_path=str(folder / f"{record.student_index}_mask.png"),
                ink_ratio=record.ink_ratio,
                components=rng.randint(1, 4),
                aspect=round(rng.uniform(1.5, 4.0), 2),
                stroke_length=rng.randint(120, 600),
            )
        written += len(records)
    return written


def clear(db: Database) -> None:
    """Empty every table, so the real pipeline starts from nothing."""
    with db.connect() as connection:
        for table in ("signatures", "attendance", "sheets", "students"):
            connection.execute(f"DELETE FROM {table}")
    print(f"cleared every table in {db.path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seed_db.py", description="fake attendance data for downstream development"
    )
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="database file to write")
    parser.add_argument("--clear", action="store_true", help="delete all rows instead of seeding")
    args = parser.parse_args(argv)

    db = Database(args.db)
    if args.clear:
        db.init_schema()
        clear(db)
        return 0

    rows = seed(db)
    print(f"seeded {rows} fake attendance rows into {db.path}")
    print("this is invented data, run tools/seed_db.py --clear before the real run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
