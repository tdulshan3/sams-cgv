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

# --- M3 enhancement ---

# --- M4 binarisation ---

# --- M5 table detection ---

# --- M6 ink segmentation ---

# --- M7 decision ---

# --- M8 recognition ---

SIG_NORM_SIZE = (220, 120)
"""Fixed (width, height) every signature is normalised to before comparison.

Two signatures cannot be compared until they sit in the same box at the same
scale — this is the box.
"""

SIG_NORM_PAD = 10
"""Pixels of white padding kept around the trimmed ink before resizing."""

HOG_ORIENTATIONS, HOG_PPC, HOG_CPB = 9, (8, 8), (2, 2)
"""HOG descriptor parameters: orientation bins, pixels per cell, cells per block."""

ORB_N_FEATURES = 500
"""Max ORB keypoints per signature."""

ORB_LOWE_RATIO = 0.75
"""Lowe's ratio test threshold for accepting an ORB keypoint match as good."""

SCORE_WEIGHTS = {"ssim": 0.30, "hog": 0.30, "hu": 0.10, "orb": 0.15, "custom": 0.15}
"""Weights for the combined score. Must sum to 1.0 — justified in T5 by which
feature separates genuine from impostor pairs best.
"""

MATCH_THRESHOLD = 0.62
"""Combined score above this is a match. Set from the EER experiment in T5 —
never guessed.
"""

UNCERTAIN_BAND = 0.08
"""Scores within this band below MATCH_THRESHOLD are reported uncertain
rather than a confident mismatch.
"""

USE_CNN_FEATURES = False
"""Tier 3, optional. Pretrained ResNet18 embedding as an extra score. Off
until T8, and only switched on if it measurably helps.
"""

# --- M9 visualisation ---


def ensure_dirs() -> None:
    """Create every output folder. Safe to call on every run."""
    for path in (OUTPUTS, STEPS, CELLS, CHARTS, FIGURES, FIXTURES):
        path.mkdir(parents=True, exist_ok=True)