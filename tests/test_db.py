"""Tests for the database and the XML parser (M7).

Both are ``src/io``, and both are the places where a small mistake is silent
rather than loud: a duplicated row still looks like data, and an index parsed
as an integer still looks like an index until the leading zero is gone. Every
test runs against a temporary database file, never ``data/attendance.db``.
"""

from __future__ import annotations

import sqlite3

import matplotlib

matplotlib.use("Agg")

import pytest

from src import config
from src.io.db import Database
from src.io.xml_parser import parse_info, parse_students
from src.models import AttendanceRecord, SheetMeta, Student

STUDENTS = [
    Student("10000409", "M S Dilshanika Perera"),
    Student("10009301", "C W M A Shehan Abeyrathne"),
]

INFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nsbm>
  <subject>
    <code>CS402.3</code>
    <name>Computer Graphics and Visualization</name>
    <lecturer>Dr. Rasika Ranaweera</lecturer>
  </subject>
  <students><batches><batch year="2016.1">
    <student><index>007</index><title>Mr</title><name>Leading Zero</name></student>
    <student><index>10000409</index><title>Ms</title><name>M S Dilshanika Perera</name></student>
  </batch></batches></students>
</nsbm>
"""


@pytest.fixture
def db(tmp_path):
    """A fresh database in a temporary folder, schema already created."""
    database = Database(tmp_path / "attendance.db")
    database.init_schema()
    return database


def sheet(date: str = "12.07.2019") -> SheetMeta:
    return SheetMeta(path=f"data/sheets/{date}.png", date=date)


def records(date: str = "12.07.2019", present: bool = True) -> list[AttendanceRecord]:
    return [
        AttendanceRecord(student.index, date, present, 0.9, 0.15) for student in STUDENTS
    ]


def test_init_schema_creates_all_four_tables(db):
    """A fresh file has students, sheets, attendance and signatures in it."""
    with db.connect() as connection:
        names = {
            row["name"]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert {"students", "sheets", "attendance", "signatures"} <= names


def test_init_schema_is_safe_to_repeat(db):
    """Every run calls it, so calling it twice must not raise or wipe anything."""
    db.upsert_students(STUDENTS)
    db.init_schema()
    assert db.get_student("10000409") is not None


def test_processing_a_sheet_twice_does_not_duplicate(db):
    """Re-running ``sams.py`` on the same sheet updates the verdict in place.

    The obvious bug here is a plain INSERT: run the sheet twice and the student
    has two attendance rows, so every chart downstream double-counts them.
    """
    db.upsert_students(STUDENTS)
    sheet_id = db.upsert_sheet(sheet())
    db.save_attendance(records(), sheet_id)

    second_id = db.upsert_sheet(sheet())
    db.save_attendance(records(present=False), second_id)

    assert second_id == sheet_id
    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM attendance").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM sheets").fetchone()[0] == 1
    assert db.get_attendance("10000409")[0]["present"] == 0


def test_get_attendance_is_in_date_order_not_string_order(db):
    """``05.07.2019`` comes *after* ``21.06.2019``, however the strings sort.

    Sheet dates are stored as they are printed, ``DD.MM.YYYY``. Sorted as text
    that puts July before June, and a chart of attendance over time then draws
    the term in the wrong order.
    """
    db.upsert_students(STUDENTS)
    for date in ("05.07.2019", "31.05.2019", "21.06.2019", "12.07.2019"):
        sheet_id = db.upsert_sheet(sheet(date))
        db.save_attendance(records(date), sheet_id)

    dates = [row["sheet_date"] for row in db.get_attendance("10000409")]
    assert dates == ["31.05.2019", "21.06.2019", "05.07.2019", "12.07.2019"]


def test_signatures_are_stored_and_returned_in_date_order(db):
    """M8 asks for a student's samples and gets them oldest first."""
    db.upsert_students(STUDENTS)
    for date in ("12.07.2019", "31.05.2019"):
        sheet_id = db.upsert_sheet(sheet(date))
        db.save_signature(
            "10000409", sheet_id, crop_path=f"outputs/cells/{date}/10000409.png", components=3
        )

    signatures = db.get_signatures("10000409")
    assert [row["sheet_date"] for row in signatures] == ["31.05.2019", "12.07.2019"]
    assert signatures[0]["components"] == 3


def test_save_signature_rejects_an_unknown_field(db):
    """A typo in a column name is a bug, not something to write nothing about."""
    db.upsert_students(STUDENTS)
    sheet_id = db.upsert_sheet(sheet())
    with pytest.raises(ValueError, match="unknown signature field"):
        db.save_signature("10000409", sheet_id, strokelength=10)


def test_sql_injection_in_an_index_changes_nothing(db):
    """Proof the queries are parameterised.

    ``infovis.py <index>`` passes whatever the user typed straight through to a
    lookup. Formatted into the SQL string rather than bound as a parameter,
    this argument would drop the table.
    """
    db.upsert_students(STUDENTS)
    sheet_id = db.upsert_sheet(sheet())
    db.save_attendance(records(), sheet_id)

    attack = "10000409'; DROP TABLE attendance; --"
    assert db.get_attendance(attack) == []
    assert db.get_student(attack) is None

    with db.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM attendance").fetchone()[0] == 2


def test_foreign_keys_are_enforced(db):
    """Attendance for a student nobody has heard of is refused, not stored."""
    sheet_id = db.upsert_sheet(sheet())
    with pytest.raises(sqlite3.IntegrityError):
        db.save_attendance([AttendanceRecord("99999999", "12.07.2019", True, 0.9, 0.2)], sheet_id)


def test_known_indices_and_is_empty_reflect_the_data(tmp_path, monkeypatch):
    """The two module-level helpers ``infovis.py`` and ``investigate.py`` call.

    Pointed at a temporary database rather than the real one, which also pins
    down that ``Database()`` reads ``config.DB_PATH`` when it is constructed.
    Bound as a default argument instead, it would freeze the path at import
    time, this test would write to ``data/attendance.db``, and the helpers
    would answer about the wrong file.
    """
    from src.io import db as db_module

    monkeypatch.setattr(config, "DB_PATH", tmp_path / "attendance.db")
    assert db_module.known_indices() == []
    assert db_module.is_empty() is True

    database = Database()
    assert database.path == tmp_path / "attendance.db"
    database.init_schema()
    database.upsert_students(STUDENTS)
    sheet_id = database.upsert_sheet(sheet())
    database.save_attendance(records(), sheet_id)

    assert db_module.known_indices() == ["10000409", "10009301"]
    assert db_module.is_empty() is False


def test_an_unreadable_database_reads_as_empty(tmp_path, monkeypatch):
    """A corrupt or half-written file must not put a traceback in front of a user.

    ``infovis.py`` asks ``is_empty()`` before it draws anything. If the file on
    disk is not a database it can read, the useful answer is "there is nothing
    here yet, run sams.py" — section 10 of the spec allows no traceback for a
    situation the user did not cause.
    """
    from src.io import db as db_module

    broken = tmp_path / "attendance.db"
    broken.write_text("this is not a database", encoding="utf-8")
    monkeypatch.setattr(config, "DB_PATH", broken)

    assert db_module.known_indices() == []
    assert db_module.is_empty() is True


def test_a_database_without_the_schema_reads_as_empty(tmp_path, monkeypatch):
    """A file that is valid SQLite but has no tables yet is also just 'empty'."""
    from src.io import db as db_module

    path = tmp_path / "attendance.db"
    sqlite3.connect(path).close()
    monkeypatch.setattr(config, "DB_PATH", path)

    assert db_module.known_indices() == []
    assert db_module.is_empty() is True


def test_parse_students_keeps_indices_as_strings(tmp_path):
    """``"007"`` survives. An index parsed as a number is a different student."""
    path = tmp_path / "info.xml"
    path.write_text(INFO_XML, encoding="utf-8")

    students = parse_students(path)

    assert [student.index for student in students] == ["007", "10000409"]
    assert all(isinstance(student.index, str) for student in students)


def test_parse_info_finds_students_at_any_depth_and_reads_the_subject(tmp_path):
    """``.//student`` survives the batch element changing shape — spec §4 deviation 4."""
    path = tmp_path / "info.xml"
    path.write_text(INFO_XML, encoding="utf-8")

    students, meta = parse_info(path)

    assert len(students) == 2
    assert meta["subject_code"] == "CS402.3"
    assert meta["lecturer"] == "Dr. Rasika Ranaweera"
    assert meta["degree"] == ""


def test_the_real_info_xml_holds_all_six_students():
    """The committed file, parsed by the committed parser."""
    students = parse_students()
    assert len(students) == 6
    assert students[0].index == "10000409"
    assert students[-1].index == "10009306"


def test_malformed_xml_gets_a_clear_error(tmp_path):
    """Figure 1 of the brief is not well-formed. Say so, do not raise ParseError."""
    path = tmp_path / "broken.xml"
    path.write_text("<nsbm><students><15><student></nsbm>", encoding="utf-8")

    with pytest.raises(ValueError, match="not well-formed XML"):
        parse_students(path)


def test_missing_xml_says_which_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="info.xml"):
        parse_students(tmp_path / "info.xml")


def test_a_student_without_an_index_is_an_error(tmp_path):
    """Silently dropping them would quietly shift every later row by one."""
    path = tmp_path / "info.xml"
    path.write_text(
        "<nsbm><students><student><name>No Index</name></student></students></nsbm>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no <index>"):
        parse_students(path)
