"""Central configuration.

Every tunable number in SAMS lives here. No module hard-codes a threshold, a
kernel size or a path — it imports the name from this file. That is what makes
the prototype adjustable without hunting through nine people's code, and it is
what lets the report state each parameter and its value in one table.

Layout: M1 owns the header and the M1 block. Every other member appends their
own block under their banner and edits nothing above it.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Everything is derived from the repository root so the project runs
# from any working directory.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SHEETS = DATA / "sheets"
FIXTURES = DATA / "fixtures"
OUTPUTS = ROOT / "outputs"
STEPS = OUTPUTS / "steps"
CELLS = OUTPUTS / "cells"
CHARTS = OUTPUTS / "charts"
FIGURES = OUTPUTS / "figures"
DB_PATH = DATA / "attendance.db"
INFO_XML = DATA / "info.xml"
GROUND_TRUTH = DATA / "ground_truth.csv"

# ---------------------------------------------------------------------------
# Shared constants. Read by more than one module.
# ---------------------------------------------------------------------------

SIGNATURE_COL = 4
"""Index of the signature column in the printed student table.

Measured in task T0. The sheet is ``No | Student No | Title | Student Name |
Signature``, so the signature is column 4. The first draft of the spec assumed
4 columns and said 3.
"""

TABLE_COLS = 5
"""Columns in the student table, header included."""

DEBUG = False
"""Verbose logging. ``sams.py --debug`` flips this through ``logging.set_debug``."""

# --- M1 lead and integration ---

SAVE_STEPS = True
"""Write each pipeline step to ``outputs/steps/<date>/`` by default."""

SHOW_PROGRESS = True
"""Open the Matplotlib montage at the end of a run by default."""

FIGURE_DPI = 150
"""Minimum dpi the brief asks for on every saved figure."""

MONTAGE_MAX_COLS = 4
"""Widest the step montage grid is allowed to get before it wraps."""

MONTAGE_FIGSIZE_PER_TILE = (3.6, 4.4)
"""Inches per montage tile, width by height. Sheets are portrait."""

STEP_IMAGE_MAX_WIDTH = 1400
"""Step images are downscaled to this width before saving.

The source photos are 3024 x 4032. Writing eight full size PNGs per sheet costs
tens of megabytes and adds seconds to every run, and nobody reads a step image
at full resolution — it goes in the report at a few inches wide.
"""

UNCERTAIN_BELOW = 0.60
"""A record with confidence under this is reported as uncertain in the summary.

It is still counted as present or absent. The line exists so a human knows how
many verdicts are worth checking by eye.
"""

# --- M2 geometry ---
TARGET_WIDTH = 1600
CANNY_LOW, CANNY_HIGH = 50, 150
MIN_SHEET_AREA_RATIO = 0.30
MAX_SKEW_CORRECTION_DEG = 15.0
BORDER_TRIM_PX = 6

# --- M3 enhancement ---

# --- M4 binarisation ---

# --- M5 table detection ---

# --- M6 ink segmentation ---

# --- M7 decision ---

# --- M8 recognition ---

# --- M9 visualisation ---


def ensure_dirs() -> None:
    """Create every output folder. Safe to call on every run."""
    for path in (OUTPUTS, STEPS, CELLS, CHARTS, FIGURES, FIXTURES):
        path.mkdir(parents=True, exist_ok=True)
