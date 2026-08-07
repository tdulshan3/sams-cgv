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
#
# Agreed final chain, tuned against Otsu binarisation on all five sheets
# (T6): luminosity greyscale -> shadow removal (kernel 25) -> bilateral
# denoise -> CLAHE contrast (clip 1.5). See BILATERAL_D / SHADOW_KERNEL /
# CLAHE_CLIP below for the reasoning behind each individual value.

GREY_METHOD = "luminosity"
"""Which of ``to_grey``'s four methods ``EnhanceStage`` uses by default."""

DENOISE_METHOD = "bilateral"
"""Which of ``denoise``'s four methods ``EnhanceStage`` uses by default."""

GAUSSIAN_KSIZE = 5
"""Kernel size (odd) for the gaussian denoise option."""

MEDIAN_KSIZE = 3
"""Kernel size (odd) for the median denoise option."""

BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE = 9, 75, 75
"""``cv2.bilateralFilter`` parameters: neighbourhood diameter, colour sigma,
space sigma."""

NLMEANS_H = 10
"""``cv2.fastNlMeansDenoising`` filter strength. OpenCV's own default of 3 is
too weak to touch the noise levels seen on the phone photos."""

SHADOW_KERNEL = 25
"""Morphological kernel size used to estimate the background lighting map."""

CONTRAST_METHOD = "clahe"
"""Which of ``enhance_contrast``'s three methods ``EnhanceStage`` uses by
default. CLAHE beats global histogram equalisation on a mostly-white page —
see T4."""

CLAHE_CLIP, CLAHE_GRID = 1.5, (8, 8)
"""``cv2.createCLAHE`` parameters: clip limit and tile grid size.

Swept 1.0-4.0 against Otsu binarisation on all five sheets: every step up in
clip limit increases both the ink percentage picked up on blank paper and the
noise std inside a blank patch, with no corresponding gain once denoise has
already run. 1.5 sits low enough on that curve to avoid amplifying paper
texture into speckle, while still lifting faint strokes above the global
default of 2.0."""

# --- M4 binarisation ---

BINARIZE_METHOD = "adaptive"
"""Which thresholding method ``BinarizeStage`` uses: ``global`` | ``otsu`` |
``adaptive`` | ``sauvola``. See the T5 comparison in ``docs/contrib_m4.md``
for the measurements behind this choice."""

THRESHOLD_GLOBAL_VALUE = 127
"""Fixed cut-off for ``threshold_global``. Kept as a deliberate failure
exhibit — see ``m4_global_failure.png`` — not as something worth tuning."""

ADAPTIVE_BLOCK, ADAPTIVE_C = 41, 12
"""``threshold_adaptive`` neighbourhood size (must be odd) and constant
subtracted from the local mean/gaussian before comparing.

Swept block 15-51 against c 5-15 on all five sheets (T3). The result is a
broad plateau rather than a sharp optimum — across the whole grid ink
coverage moves only between 7.6% and 10.0% — so these two numbers are
chosen to avoid the edges of that plateau rather than to chase a peak:

- ``c = 5`` is a cliff, not a slope: component count jumps from ~400 to
  ~1100 as paper texture starts crossing the threshold. Anything from 8
  upward is stable, and 12 sits comfortably clear of the cliff.
- Larger blocks preserve the thin printed table lines slightly better,
  because a thin dark line contrasts more strongly against a wider bright
  neighbourhood. Below block 21 line survival starts dropping. 41 keeps
  that margin without drifting so wide that the threshold stops being
  local and starts behaving globally.
"""

SAUVOLA_WINDOW = 25
"""``threshold_sauvola`` local neighbourhood size, must be odd."""

SAUVOLA_K, SAUVOLA_R = 0.2, 128.0
"""Sauvola's ``k`` (how strongly local contrast pulls the threshold away from
the local mean) and ``r`` (the dynamic range of the data).

``r`` must be passed explicitly. Left to infer it, scikit-image takes it from
the array's dtype limits, and for a float array those are ``(-1, 1)`` — so
``r`` becomes 1.0 rather than ~128, the local threshold lands around 1000 on
0-255 data, and every pixel falls below it. The whole page comes out as ink."""

MORPH_OPEN_K, MORPH_CLOSE_K = 2, 3
"""Opening and closing kernel sizes for ``morph_clean``.

Opening removes speckle, closing repairs broken pen strokes. Both are kept
deliberately small, and the sweep over all five sheets (T6) shows why. As
the closing kernel grows, component count collapses while ink coverage
climbs:

    open/close   ink %   components
        2 / 3     8.78          381
        2 / 5     8.96          287
        3 / 7     9.38          174

Falling components with *rising* ink is not cleaning — it is separate
objects being welded into one. By kernel 7 more than half the components on
the page have merged into a neighbour, and on a signing sheet the nearest
neighbour of a signature is the printed table border it sits against. M6
would then measure a signature that is partly table line. At 2/3 the
component count drops (402 -> 381) with ink essentially unchanged, which is
speckle genuinely being removed rather than strokes being fused."""

SIG_MORPH_OPEN_K, SIG_MORPH_CLOSE_K = 2, 2
"""Kernel sizes for ``clean_signature_crop``, M8's small-crop variant.

Smaller than the whole-sheet pair above. The sheet kernels are sized against
a 1600-pixel-wide page; applied to a signature crop a couple of hundred
pixels across they erode a thin ballpoint stroke away entirely."""

MORPH_KERNEL_SHAPE = "ellipse"
"""Structuring element shape: ``ellipse`` | ``rect`` | ``cross``.

Ellipse approximates the rounded disc a ballpoint actually lays down, so it
erodes strokes evenly rather than squaring off their ends the way a
rectangle does."""

# --- M5 table detection ---

H_KERNEL_RATIO = 0.30
"""Horizontal morphology kernel width as a fraction of image width.
Only runs longer than this fraction survive as horizontal lines."""

V_KERNEL_RATIO = 0.30
"""Vertical morphology kernel height as a fraction of image height."""

LINE_MERGE_TOL = 8
"""Peaks within this many pixels are merged into a single line position."""

MIN_ROW_HEIGHT = 18
"""Rows separated by fewer pixels than this are treated as duplicates."""

MIN_COL_WIDTH = 25
"""Columns separated by fewer pixels than this are treated as duplicates."""

CELL_INSET = 4
"""Pixels shaved off each side of a cell crop to exclude the border line."""

CELL_PAD_Y = 6
"""Extra pixels added below a signature cell crop to catch overflowing strokes.

On 31.05.2019 and 05.07.2019 signatures cross into the row below."""

EXPECTED_COLS = 5
"""No | Student No | Title | Student Name | Signature (measured T0)."""

EXPECTED_DATA_ROWS = 6
"""Every sheet has exactly 6 student rows (measured T0)."""


# --- M6 ink segmentation ---

# --- M7 decision ---

# --- M8 recognition ---

# --- M9 visualisation ---


def ensure_dirs() -> None:
    """Create every output folder. Safe to call on every run."""
    for path in (OUTPUTS, STEPS, CELLS, CHARTS, FIGURES, FIXTURES):
        path.mkdir(parents=True, exist_ok=True)
