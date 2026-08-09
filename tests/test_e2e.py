import pytest
import sqlite3
import subprocess
import sys
import os
from pathlib import Path

from src import config
from src.io.db import Database


@pytest.fixture
def clean_db():
    """Ensure the database is clean before and after testing."""
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    
    yield config.DB_PATH
    
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()


def test_pipeline_writes_expected_rows(clean_db):
    """The full pipeline on one sheet writes the expected number of attendance rows."""
    # We pick the first available sheet
    sheets = list(config.SHEETS.glob("*.png"))
    if not sheets:
        pytest.skip("No sheet images found in data/sheets/")
        
    sheet_path = sheets[0]
    
    # Run the pipeline via CLI
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "sams.py", str(sheet_path), str(config.INFO_XML), "--no-show", "--no-save"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    assert result.returncode == 0, f"sams.py failed: {result.stderr}"
    
    db = Database()
    with db.connect() as conn:
        count = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]
        
    # We expect 6 rows because there are 6 students on each sheet
    assert count == 6


def test_running_twice_leaves_count_unchanged(clean_db):
    """Running the same sheet twice leaves the row count unchanged."""
    sheets = list(config.SHEETS.glob("*.png"))
    if not sheets:
        pytest.skip("No sheet images found in data/sheets/")
        
    sheet_path = sheets[0]
    cmd = [sys.executable, "sams.py", str(sheet_path), str(config.INFO_XML), "--no-show", "--no-save"]
    
    # Run first time
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    
    db = Database()
    with db.connect() as conn:
        count_first = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]
        
    assert count_first > 0
    
    # Run second time
    subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    
    with db.connect() as conn:
        count_second = conn.execute("SELECT COUNT(*) as c FROM attendance").fetchone()["c"]
        
    assert count_second == count_first
