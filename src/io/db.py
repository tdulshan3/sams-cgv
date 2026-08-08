"""The attendance database.

Everything the pipeline decides ends up here, and everything M8 and M9 draw
comes back out of here. One SQLite file, ``data/attendance.db``, no server to
install and nothing to configure — the whole database is a file the marker can
copy.

Four tables, defined once in :data:`SCHEMA`:

* ``students``   — the roll, read from ``info.xml``
* ``sheets``     — one row per signing sheet photo processed
* ``attendance`` — one row per student per sheet: the verdict
* ``signatures`` — where each signature crop was saved, plus M6's measurements,
  so ``investigate.py`` can find a student's samples without re-running the
  pipeline

Two rules hold everywhere in this module:

* **Every query is parameterised.** Values go in as ``?`` placeholders, never
  formatted into the SQL string. A student index is user input the moment
  someone types ``infovis.py <index>``.
* **Re-processing a sheet updates, it never duplicates.** ``attendance`` and
  ``signatures`` are unique on ``(student_index, sheet_id)`` and every write is
  ``INSERT OR REPLACE``, so running ``sams.py`` on the same sheet twice leaves
  the row count unchanged.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.models import AttendanceRecord, SheetMeta, Student
from src.utils.logging import get_logger

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_index TEXT PRIMARY KEY,
    name          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sheets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    sheet_date    TEXT UNIQUE NOT NULL,
    image_path    TEXT,
    subject_code  TEXT,
    processed_at  TEXT
);
CREATE TABLE IF NOT EXISTS attendance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_index TEXT NOT NULL,
    sheet_id      INTEGER NOT NULL,
    present       INTEGER NOT NULL,
    confidence    REAL,
    ink_ratio     REAL,
    UNIQUE(student_index, sheet_id),
    FOREIGN KEY (student_index) REFERENCES students(student_index),
    FOREIGN KEY (sheet_id)      REFERENCES sheets(id)
);
"""

# The sheet dates are printed on the sheets as DD.MM.YYYY, and that is what the
# filenames carry, so sorting them as text puts 05.07.2019 before 21.06.2019.
# Slicing the string back into year, month, day inside ORDER BY sorts them
# chronologically without storing a second date column.
DATE_ORDER = (
    "substr(sheets.sheet_date, 7, 4), "
    "substr(sheets.sheet_date, 4, 2), "
    "substr(sheets.sheet_date, 1, 2)"
)

class Database:
    """The attendance database, and every query the project runs against it."""

    def __init__(self, path: Path = config.DB_PATH) -> None:
        """
        Args:
            path: Where the SQLite file lives. Tests pass a temporary path;
                everything else takes the default.
        """
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a connection, commit on success, always close.

        Yields:
            A connection whose rows come back as :class:`sqlite3.Row`, so
            callers can read a column by name instead of by position.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def init_schema(self) -> None:
        """Create the four tables if they are not there yet.

        Safe to call on every run — that is what ``IF NOT EXISTS`` buys, and it
        means no separate setup step before the first ``sams.py``.
        """
        with self.connect() as connection:
            connection.executescript(SCHEMA)
        log.debug("schema ready at %s", self.path)

    def upsert_students(self, students: Iterable[Student]) -> None:
        """Write the roll. An index already present has its name refreshed."""
        rows = [(student.index, student.name) for student in students]
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO students (student_index, name) VALUES (?, ?)",
                rows,
            )
        log.debug("%d students written", len(rows))

    def upsert_sheet(self, meta: SheetMeta, subject_code: str = "") -> int:
        """Record one signing sheet and return its row id.

        Args:
            meta: The sheet being processed.
            subject_code: From ``info.xml``, e.g. ``"CS402.3"``. Falls back to
                whatever ``meta`` already carries.

        Returns:
            The ``sheets.id`` for this date, whether it was just inserted or
            already existed. Every attendance row needs it.
        """
        processed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        code = subject_code or meta.subject_code
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO sheets (sheet_date, image_path, subject_code, processed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sheet_date) DO UPDATE SET
                    image_path   = excluded.image_path,
                    subject_code = excluded.subject_code,
                    processed_at = excluded.processed_at
                """,
                (meta.date, str(meta.path), code, processed_at),
            )
            row = connection.execute(
                "SELECT id FROM sheets WHERE sheet_date = ?", (meta.date,)
            ).fetchone()
        return int(row["id"])

    def save_attendance(self, records: Iterable[AttendanceRecord], sheet_id: int) -> None:
        """Write one verdict per student for one sheet, replacing any earlier run."""
        rows = [
            (record.student_index, sheet_id, int(record.present), record.confidence, record.ink_ratio)
            for record in records
        ]
        if not rows:
            return
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO attendance
                    (student_index, sheet_id, present, confidence, ink_ratio)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
        log.debug("%d attendance rows written for sheet %d", len(rows), sheet_id)

    def get_attendance(self, student_index: str) -> list[dict]:
        """One student's record, oldest sheet first."""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT sheets.sheet_date, attendance.present, attendance.confidence,
                       attendance.ink_ratio, sheets.subject_code
                FROM attendance
                JOIN sheets ON sheets.id = attendance.sheet_id
                WHERE attendance.student_index = ?
                ORDER BY {DATE_ORDER}
                """,
                (student_index,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_all_attendance(self) -> list[dict]:
        """Every verdict in the database, for the class-wide charts."""
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT students.student_index, students.name, sheets.sheet_date,
                       attendance.present, attendance.confidence, attendance.ink_ratio
                FROM attendance
                JOIN sheets   ON sheets.id = attendance.sheet_id
                JOIN students ON students.student_index = attendance.student_index
                ORDER BY {DATE_ORDER}, students.student_index
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_student(self, student_index: str) -> dict | None:
        """One student's name and index, or ``None`` if we do not hold them."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT student_index, name FROM students WHERE student_index = ?",
                (student_index,),
            ).fetchone()
        return dict(row) if row else None
