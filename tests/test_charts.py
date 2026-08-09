import matplotlib
matplotlib.use("Agg")  # Must be before pyplot import

import pytest
from pathlib import Path
from matplotlib.figure import Figure
import sqlite3

from src.io.db import Database
from src.viz.charts import AttendanceCharts
from src.models import Student, SheetMeta, AttendanceRecord


@pytest.fixture
def db(tmp_path):
    """Provide a fresh database for testing."""
    db_file = tmp_path / "test_attendance.db"
    d = Database(db_file)
    d.init_schema()
    return d


@pytest.fixture
def populated_db(db):
    """Provide a database with known test records."""
    db.upsert_students([
        Student("10000001", "Alice"),
        Student("10000002", "Bob"),
    ])
    
    # Insert sheets out of chronological string order, but chronologically correct
    sheet1 = db.upsert_sheet(SheetMeta(Path("21.06.2019.png"), "21.06.2019"))
    sheet2 = db.upsert_sheet(SheetMeta(Path("05.07.2019.png"), "05.07.2019"))
    
    db.save_attendance([
        AttendanceRecord("10000001", "21.06.2019", True, 0.9),
        AttendanceRecord("10000002", "21.06.2019", False, 0.9),
    ], sheet1)
    
    db.save_attendance([
        AttendanceRecord("10000001", "05.07.2019", True, 0.9),
        AttendanceRecord("10000002", "05.07.2019", True, 0.9),
    ], sheet2)
    
    return db


def test_charts_return_figure(populated_db):
    """Each chart function returns a matplotlib.figure.Figure."""
    charts = AttendanceCharts(populated_db)
    
    assert isinstance(charts.student_timeline("10000001"), Figure)
    assert isinstance(charts.student_summary("10000001"), Figure)
    assert isinstance(charts.student_dashboard("10000001"), Figure)
    assert isinstance(charts.class_heatmap(), Figure)
    assert isinstance(charts.sheet_totals(), Figure)
    assert isinstance(charts.attendance_distribution(), Figure)


def test_empty_state_no_crash(db):
    """A student with no records produces an empty-state chart, not a crash."""
    charts = AttendanceCharts(db)
    
    # Call all methods and verify they return a Figure without crashing
    assert isinstance(charts.student_timeline("99999999"), Figure)
    assert isinstance(charts.student_summary("99999999"), Figure)
    assert isinstance(charts.student_dashboard("99999999"), Figure)
    assert isinstance(charts.class_heatmap(), Figure)
    assert isinstance(charts.sheet_totals(), Figure)
    assert isinstance(charts.attendance_distribution(), Figure)


def test_dates_ordered_chronologically(populated_db):
    """Dates are ordered chronologically — assert 05.07.2019 comes after 21.06.2019."""
    records = populated_db.get_attendance("10000001")
    
    assert len(records) == 2
    assert records[0]["sheet_date"] == "21.06.2019"
    assert records[1]["sheet_date"] == "05.07.2019"


def test_save_writes_file(populated_db, tmp_path, monkeypatch):
    """save() writes a file of non-zero size."""
    import src.config
    # Redirect CHARTS output to tmp_path
    monkeypatch.setattr(src.config, "CHARTS", tmp_path)
    
    charts = AttendanceCharts(populated_db)
    fig = charts.student_summary("10000001")
    
    out_path = charts.save(fig, "test_summary")
    
    assert out_path.exists()
    assert out_path.stat().st_size > 0
