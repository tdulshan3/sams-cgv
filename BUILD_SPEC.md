# BUILD_SPEC.md — SAMS (Student Attendance Management System)

**Repository root file. This is the single source of truth for the whole group.**
Owner of this file: **M1 (Lead / Integration)**. Nobody else edits it — they raise a change request.

Module: CS402.3 Computer Graphics and Visualization, NSBM Green University
Coursework weight: 20% — Prototype 60%, Report 25%, Individual contribution 15%
Group: 9 members, M1 to M9, one module each

| | Member | Module | Owns |
|---|---|---|---|
| M1 | T.R.D.T. Dulshan | Lead / Integration & Progress Viewer | skeleton, contracts, pipeline, 3 CLIs, step viewer |
| M2 | | Acquisition & Geometry | load the photo, flatten it |
| M3 | | Greyscale & Enhancement | shadow removal, denoise, contrast |
| M4 | | Binarisation & Morphology | ink 255 / paper 0, clean it |
| M5 | | Table Detection | find the lines, cut out the cells |
| M6 | | Ink Segmentation | multi-colour pen ink masks and features |
| M7 | | Decision & Database | present or absent, info.xml, SQLite |
| M8 | | Signature Recognition | `investigate.py`, the bonus-mark module |
| M9 | | Visualisation & QA | `infovis.py` charts, end-to-end testing |

Fill in the names before submission — the front page of the report needs them.

---

## 0. How to use this file

1. Read **§0 to §8** in full before writing any code. Those sections are shared by everyone and are not negotiable.
2. Find your own module in **§9**. Implement **only** that. Every other module belongs to another person — if you need something changed there, message them, do not edit their file.
3. Work task by task, in order. Commit after each task with the exact message given (§12).
4. After each task, run the verification listed for it. Do not move on if it fails.
5. If reality contradicts this spec, **tell M1**. M1 updates this file, commits that change, and only then does anyone write code against it. §4 is the record of every time that has already happened.
6. Until your module lands, a placeholder in `src/stubs.py` stands in for it, so `sams.py` runs end to end for everyone from day one (§8). When you merge, M1 deletes your stub and wires you into `STAGES`.

**Your brief document is not this file.** Each member was given an `M<N>_*.md` brief. Those were written before anyone had looked at the real sheets, and several of their assumptions are wrong — the sheet has 5 columns and not 4, indices are 8 digits and not 3, there are two tables on the page. **Where your brief and this file disagree, this file wins.** §4 lists every correction, and §9 repeats the ones that affect each module.

---

## 1. Goal

Take a phone photo of a paper attendance signing sheet, work out who signed and who did not, store the result in a local database, and visualise it.

Three commands must work. These are fixed by the coursework brief and must never change:

```bash
python sams.py data/sheets/10.07.2019.png data/info.xml
python infovis.py 001
python investigate.py 001
```

The *shape* of those three commands is fixed. The arguments in them are the brief's illustration, not our data: T0 measured that our five sheets are dated `31.05.2019`, `21.06.2019`, `28.06.2019`, `05.07.2019`, `12.07.2019` and that student indices are 8 digits. The equivalent real invocations are:

```bash
python sams.py data/sheets/12.07.2019.png data/info.xml
python infovis.py 10000409
python investigate.py 10000409
```

M1 delivers: the repository skeleton, shared contracts, the pipeline runner, all three CLI programs, the step-by-step progress viewer, and stubs for every other module so the pipeline runs end to end from the first hour.

---

## 2. Non-negotiables

- **CLI only.** No web app, no desktop GUI, no Streamlit, no Flask. It earns no marks and costs time.
- **Ink is 255 (white). Paper is 0 (black).** Every module assumes this.
- **Student indices are strings.** `"007"`, never `7`. Leading zeros must survive.
- **No tunable number is hard-coded in a module.** All constants live in `src/config.py`.
- **Every stage subclasses `Stage`.** OOP quality is explicitly marked.
- **No file is written outside `outputs/` and `data/attendance.db`.**
- Python **3.11+**.
- The pipeline must **never silently swallow an error**. Log the failing stage name and re-raise.

---

## 3. Environment

Development machine is Arch-based Linux (CachyOS), fish shell.

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

`requirements.txt` (unpinned during development, pinned at T10):

```
opencv-python
numpy
scikit-image
scipy
matplotlib
seaborn
pandas
scikit-learn
pytest
Pillow
```

Notes:
- Do **not** `pip install` into the system Python on Arch — it is externally managed. Always use the venv.
- `opencv-python` (not `opencv-python-headless`) — we need `cv2.imshow` availability even though Matplotlib is the primary display.
- Matplotlib backend: use the default interactive backend for normal runs, and `Agg` inside tests (`matplotlib.use("Agg")` at the top of test files).

---

## 4. Input data — verify before coding

The five sheet photos come from `CGV Signing Sheets.zip`. They arrived as `1.jpeg` … `5.jpeg`, so T0 renamed each to the date printed on the sheet and converted it to lossless PNG:

```
data/sheets/31.05.2019.png    # was 1.jpeg
data/sheets/21.06.2019.png    # was 2.jpeg
data/sheets/28.06.2019.png    # was 3.jpeg
data/sheets/05.07.2019.png    # was 4.jpeg
data/sheets/12.07.2019.png    # was 5.jpeg
data/info.xml
data/ground_truth.csv         # transcribed by eye, used by M7 and M9 for accuracy
```

`info.xml` was **not** in the material we received. It is reconstructed from Figure 1 of the brief plus the printed student table on the sheets — see the deviations below.

**Measured facts (T0, `python tools/inspect_inputs.py`):**

| Fact | Value | How to get it |
|---|---|---|
| Number of sheets | 5 | `ls data/sheets` |
| Image resolution (per sheet) | 3024 x 4032 portrait, all five identical | `tools/inspect_inputs.py` |
| Colour or greyscale photos | Colour, 3 channel uint8 RGB (iPhone 7, iOS 12.1.4) | channel check |
| EXIF orientation present | No — tag 274 absent, pixels already upright | Pillow `_getexif()` |
| Table columns on the sheet | **5** — `No`, `Student No`, `Title`, `Student Name`, `Signature` | look at the image |
| Data rows per sheet | 6 on every sheet | count by eye |
| Header row present | Yes, one, printed in bold | look |
| `info.xml` root tag | `nsbm` | `head data/info.xml` |
| `info.xml` student tag path | `nsbm/students/batches/batch/student` with `index`, `title`, `name` | read it |
| Index format in XML | 8 digits, e.g. `10000409` — not `001` | inspection report |
| Do sheet rows match XML order? | Yes — the XML was transcribed in sheet row order | compare |

**Deviations from the assumptions this spec was written under. Every one of these is load-bearing:**

1. **The signature column is index 4, not 3.** The sheet has a `Title` column (`Mr` / `Ms`) between the student number and the name. `SIGNATURE_COL = 4` in §6.4, and `Cell.col` in §6.1 means `0=No, 1=Student No, 2=Title, 3=Student Name, 4=Signature`.
2. **There are two tables on each sheet.** A one-row lecture header table (`Date | time | Lecture's Name | Signatue`) sits directly above the student table. M5 must take the **lower, taller** table and ignore the header table, otherwise the lecturer's own signature is read as a student's.
3. **Indices are 8 digits, not 3.** They are still strings — `"10000409"` — and `infovis.py 001` from the brief has no matching student in our data. The CLIs accept any index string and report unknown ones helpfully.
4. **The brief's `info.xml` is not well-formed.** Figure 1 shows `<15>` as the batch element. XML tag names may not start with a digit, so no standard parser will read that document. We carry the batch as `<batch year="2016.1">`. M7's parser should locate students with `.//student` so it survives either shape.
5. **`info.xml` carries a `<title>` element** per student, matching the sheet's Title column. Figure 1 has only `index` and `name`.
6. **Signature strokes routinely cross cell borders.** On `31.05.2019` and `05.07.2019` a signature spills a long way into the row below. M5's cells and M6's ink masks cannot assume a signature is contained by its box.
7. **Ink is not always a signature.** On `21.06.2019` the last cell holds the lecturer's handwritten `ab` (absent); on `05.07.2019` one cell holds a stray red pen tick. Both are ink and both mean *absent*. Pure ink-ratio thresholding gets these wrong — this is exactly the discussion M7's decision stage must handle and the report must cover.
8. **Six students, five sheets = 30 records** in total. Small enough that the summary table in §7 shows single digit counts, not the 42 in the illustration.

Once measured, **replace the TBDs in this file and commit** before writing pipeline code. M5 and M7 depend on these answers.

---

## 5. Repository layout

```
sams-cgv/
├── BUILD_SPEC.md              # this file — M1
├── README.md                  # M1
├── requirements.txt           # M1
├── .gitignore                 # M1
├── sams.py                    # M1
├── infovis.py                 # M1
├── investigate.py             # M1
├── src/
│   ├── __init__.py
│   ├── config.py              # M1 owns the file; each member adds a commented block
│   ├── models.py              # M1
│   ├── pipeline.py            # M1
│   ├── cli.py                 # M1 — shared helpers for the three programs
│   ├── stubs.py               # M1 — placeholder stages, deleted stage by stage
│   ├── utils/
│   │   ├── stage.py           # M1
│   │   ├── logging.py         # M1
│   │   ├── timing.py          # M1
│   │   └── cvcompat.py        # M1 — opencv 5 return-shape wrappers (§6.6)
│   ├── io/
│   │   ├── image_loader.py    # M2
│   │   ├── xml_parser.py      # M7
│   │   └── db.py              # M7
│   ├── preprocess/
│   │   ├── deskew.py          # M2
│   │   ├── enhance.py         # M3
│   │   └── binarize.py        # M4
│   ├── table/
│   │   ├── line_detect.py     # M5
│   │   ├── grid_builder.py    # M5
│   │   └── cell_extract.py    # M5
│   ├── detect/
│   │   ├── cell_clean.py      # M6
│   │   ├── ink_mask.py        # M6
│   │   └── presence.py        # M7
│   ├── recognise/
│   │   ├── preprocess_sig.py  # M8
│   │   ├── features.py        # M8
│   │   ├── matcher.py         # M8
│   │   └── report.py          # M8
│   └── viz/
│       ├── progress.py        # M1
│       ├── charts.py          # M9
│       └── style.py           # M9
├── tools/
│   ├── inspect_inputs.py      # M1
│   ├── make_fixtures.py       # M1
│   ├── make_m1_figures.py     # M1 — the three figures in §13
│   ├── seed_db.py             # M7
│   ├── eval_recognition.py    # M8 — the genuine vs impostor experiment
│   ├── run_all_sheets.py      # M9
│   └── make_report_assets.py  # M9
├── tests/
│   ├── test_pipeline.py       # M1
│   ├── test_geometry.py       # M2
│   ├── test_enhance.py        # M3
│   ├── test_binarize.py       # M4
│   ├── test_table.py          # M5
│   ├── test_ink.py            # M6
│   ├── test_decision.py  test_db.py       # M7
│   ├── test_recognition.py    # M8
│   └── test_charts.py  test_e2e.py        # M9
├── data/
│   ├── sheets/  info.xml  fixtures/  ground_truth.csv  attendance.db
├── outputs/
│   ├── steps/  cells/  charts/  figures/
└── docs/
    └── contrib_m1.md … contrib_m9.md
```

`.gitignore` must contain: `.venv/`, `__pycache__/`, `*.pyc`, `outputs/`, `data/attendance.db`, `data/fixtures/`, `.pytest_cache/`.

**`data/sheets/` and `data/info.xml` ARE committed** — the marker needs them to run the prototype.

---

## 6. Shared contracts

### 6.1 `src/models.py`

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Student:
    index: str                              # "001" — always a string
    name: str
    row: int | None = None                  # sheet row, filled by M7

@dataclass
class SheetMeta:
    path: Path
    date: str                               # from the filename, e.g. "10.07.2019"
    subject_code: str = ""

@dataclass
class Cell:
    row: int                                # 0 = first DATA row, header excluded
    col: int                                # 0=No, 1=index, 2=title, 3=name, 4=signature
    bbox: tuple[int, int, int, int]         # x, y, w, h in the warped image
    image: np.ndarray | None = None         # BGR crop
    student_index: str | None = None

@dataclass
class InkResult:
    cell: Cell
    mask: np.ndarray | None = None          # uint8, ink = 255
    ink_ratio: float = 0.0
    components: int = 0
    stroke_bbox: tuple[int, int, int, int] | None = None
    aspect: float = 0.0
    stroke_length: int = 0
    crop_path: str | None = None
    mask_path: str | None = None

@dataclass
class AttendanceRecord:
    student_index: str
    sheet_date: str
    present: bool
    confidence: float
    ink_ratio: float = 0.0
```

### 6.2 `src/utils/stage.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class Stage(ABC):
    """One step of the image pipeline."""
    name: str = "stage"

    @abstractmethod
    def run(self, ctx: dict) -> dict:
        """Read from ctx, write results back into ctx, return ctx."""

    def figures(self) -> dict[str, np.ndarray]:
        """Images this stage wants shown and saved. Default: none."""
        return {}
```

### 6.3 The context dictionary

Single object passed down the line. Keys are fixed.

| key | type | written by | read by |
|---|---|---|---|
| `sheet` | `SheetMeta` | M1 | all |
| `students` | `list[Student]` | M1 (via M7 parser) | M7 |
| `bgr` | BGR uint8 | M2 | M2 |
| `warped` | BGR uint8, flattened sheet | M2 | M3, M5 |
| `grey` | uint8, 1 channel | M3 | M4 |
| `binary` | uint8, ink=255 | M4 | M5 |
| `grid` | `Grid` | M5 | M5 |
| `cells` | `list[Cell]`, signature column, `.image` set from `warped` | M5 | M6 |
| `ink` | `list[InkResult]`, same order as `cells` | M6 | M7 |
| `records` | `list[AttendanceRecord]` | M7 | M1 |

**Stage order is fixed:**
`geometry → enhance → binarize → table → ink → decision`

### 6.4 `src/config.py`

M1 writes the header and the M1 block. Each member appends their own block under a comment banner. Nobody edits another member's block.

```python
from pathlib import Path

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

SIGNATURE_COL = 4          # measured in T0: the sheet has a Title column
SAVE_STEPS = True
SHOW_PROGRESS = True
FIGURE_DPI = 150

def ensure_dirs() -> None:
    for p in (OUTPUTS, STEPS, CELLS, CHARTS, FIGURES, FIXTURES):
        p.mkdir(parents=True, exist_ok=True)

# --- M2 geometry ---
# --- M3 enhancement ---
# --- M4 binarisation ---
# --- M5 table detection ---
# --- M6 ink segmentation ---
# --- M7 decision ---
# --- M8 recognition ---
# --- M9 visualisation ---
```

### 6.5 Entry points the CLIs call

The three programs are thin wrappers. They must not know how anything works, only what to call. These are the functions M1's CLIs import; until a module lands, `src/cli.py` catches the `ImportError` and prints *module not ready yet*.

| Function | Owner | Returns |
|---|---|---|
| `src.io.xml_parser.parse_students(path: Path) -> list[Student]` | M7 | the roll, in sheet row order |
| `src.io.db.known_indices() -> list[str]` | M7 | every student index the database has a record for |
| `src.io.db.is_empty() -> bool` | M7 | `True` when no attendance has been stored yet |
| `src.viz.charts.show_student(index: str, save_only: bool) -> None` | M9 | draws one student's attendance |
| `src.viz.charts.show_all(save_only: bool) -> None` | M9 | draws the whole class |
| `src.recognise.matcher.investigate(index: str, save_only: bool) -> None` | M8 | compares that student's signatures and reports mismatches |

`save_only=True` means write to `outputs/` and open no window — needed so the whole prototype can be run over SSH or in a test.

### 6.6 `src/utils/cvcompat.py` — OpenCV 5, read this before using Hough

We pin **opencv-python 5**. Several OpenCV functions that returned `(N, 1, K)` in version 4 return `(N, K)` in version 5, and essentially every tutorial, Stack Overflow answer and textbook example online was written against version 4. Code copied from them looks correct, passes a synthetic unit test, and dies on the first real photo:

```
TypeError: cannot unpack non-iterable numpy.int32 object
```

**This has already cost the group two modules** — M2 in PR #3 and M5 in PR #11 — and both times it was found only by running `sams.py` on a real sheet, never by the test suite.

So: **never call `cv2.HoughLinesP` directly.** Call the wrapper, which returns `(N, 4)` on every version and an empty array rather than `None`:

```python
from src.utils.cvcompat import hough_line_segments

for x1, y1, x2, y2 in hough_line_segments(binary, threshold=80):
    ...
```

`tests/test_pipeline.py::test_nobody_calls_houghlinesp_directly` fails the build if any module under `src/` reaches past it. If you hit the same shape change on another OpenCV call (`findContours`, `goodFeaturesToTrack`, ORB output), add a wrapper here rather than a fix in your own module — that is what stops it happening a third time.

### 6.7 `src/cli.py`

M1 owns a small shared helper module for the three programs: index validation, the *module not ready* message, and the friendly-error-then-`exit(2)` path. It exists so the same 20 lines are not written three times, and so all three commands fail in exactly the same way.

---

## 7. CLI specifications

### `sams.py`

```
usage: sams.py [-h] [--no-show] [--no-save] [--debug] image xml

positional:
  image          path to a signing sheet photo
  xml            path to info.xml

options:
  --no-show      do not open the montage window
  --no-save      do not write step images
  --debug        verbose logging
```

Behaviour:
1. `config.ensure_dirs()`
2. Validate both paths. Missing file → print a clear message, `sys.exit(2)`. No traceback.
3. Derive `sheet_date` from the image filename stem.
4. Parse `info.xml` → `ctx["students"]` (via M7's parser; stub returns an empty list).
5. Build the stage list in the fixed order and run the `Pipeline`.
6. Print the summary table.
7. `viewer.save_all()` then `viewer.show_montage()` unless suppressed.

Required final output:

```
─────────────────────────────────────────────────────────────
 Sheet     : 10.07.2019
 Students  : 42
 Present   : 38
 Absent    : 4
 Uncertain : 1
 Duration  : 6.24 s
 Steps     : outputs/steps/10.07.2019/  (8 images)
─────────────────────────────────────────────────────────────
```

### `infovis.py`

```
usage: infovis.py [-h] [--all] [--save-only] [index]
```
Unknown index → print a message plus up to 10 valid indices, `sys.exit(2)`. Empty database → tell the user to run `sams.py` first. Delegates to `src/viz/charts.py` (M9).

### `investigate.py`

```
usage: investigate.py [-h] [--save-only] index
```
Fewer than 2 signature samples → clear message, exit 0 (not an error). Delegates to `src/recognise/matcher.py` (M8).

A *signature sample* means a saved crop at `outputs/cells/<sheet_date>/<index>.png` (§5.4). Counting files needs no database and no other module, so `investigate.py` can give a straight answer before M7 or M8 exist.

---

## 8. Stub policy

M1 creates `src/stubs.py` containing a placeholder `Stage` subclass for **every** module not yet written, so `sams.py` runs end to end immediately.

Rules for stubs:
- Each stub logs `STUB <name>: returning placeholder data`.
- Each stub writes a **type-correct** value into `ctx` (an empty list, a grey copy, whatever the contract says) so downstream code does not crash.
- Stubs read from `data/fixtures/` where an image is needed. The one exception is `bgr`: `GeometryStub` loads the photo the user actually passed on the command line, because faking file loading would make `sams.py <any sheet>` show the same picture for all five sheets and hide real problems such as an unreadable file.
- `data/fixtures/` is not committed. A stub that needs a fixture and cannot find one falls back to something type-correct, logs a `WARNING` naming `tools/make_fixtures.py`, and keeps the run alive. A fresh clone must reach the summary table without running any tool first.
- Every stub carries `# STUB — owned by M<N>, delete when their module lands`.
- A stub is deleted the moment the real module is merged. Stubs must not survive to submission — grep for `STUB` before tagging the release.

| Stub | Writes to ctx |
|---|---|
| `GeometryStub` | `bgr` ← the real photo at `sheet.path`; `warped` ← `fixtures/warped.png` |
| `EnhanceStub` | `grey` ← `cvtColor(warped, BGR2GRAY)` |
| `BinarizeStub` | `binary` ← Otsu on `grey`, inverted |
| `TableStub` | `grid=None`, `cells=[]` |
| `InkStub` | `ink=[]` |
| `DecisionStub` | `records=[]` |

---

## 9. Task lists

One subsection per member. Work through your own in order — implement, verify, commit — and ignore the rest except to know what arrives from the person before you.

Every module, without exception:

- Subclasses `Stage` for its pipeline step (§6.2) and honours the context keys in §6.3.
- Puts its tunable numbers in its own block in `src/config.py` and hard-codes nothing.
- Writes its tests in `tests/test_<area>.py` (§11) and its figures to `outputs/figures/m<N>_*.png` at 150 dpi minimum.
- Drafts `docs/contrib_m<N>.md` as it goes, not the night before.
- Reaches 15+ commits on branch `feat/m<N>-<area>` (§12).

---

## 9.1 M1 — Lead / Integration & Progress Viewer

**Branch** `feat/m1-core` · **Owns** `sams.py`, `infovis.py`, `investigate.py`, `src/config.py`, `src/models.py`, `src/pipeline.py`, `src/cli.py`, `src/stubs.py`, `src/utils/*`, `src/viz/progress.py`, `tools/inspect_inputs.py`, `tools/make_fixtures.py`, `tools/make_m1_figures.py`, `tests/test_pipeline.py`

Work in this order. Each task: implement → verify → commit.

---

### T0 — Inspect the real input data
Write `tools/inspect_inputs.py`. It prints per sheet: filename, resolution, channel count, file size, EXIF orientation tag. Then it prints the first 40 lines of `info.xml` and the tag structure found. Run it and **fill in the §4 table in this file**.

**Verify:** `python tools/inspect_inputs.py` prints one block per sheet and the XML structure.
**Commits:**
- `chore(tools): add input inspection script for sheets and xml`
- `docs(spec): record measured input data facts in build spec`

---

### T1 — Repository skeleton  *(blocking — nobody else starts before this)*
Full folder tree from §5, all `__init__.py` files, `requirements.txt`, `.gitignore`, a placeholder `README.md`. Commit the sheet images and `info.xml` into `data/`.

**Verify:** `python -c "import src"` succeeds; `git status` is clean; `outputs/` is ignored.
**Commits:**
- `chore(repo): add project skeleton and package structure`
- `chore(repo): add requirements and gitignore`
- `chore(data): add signing sheet images and info.xml`

---

### T2 — Shared contracts  *(blocking)*
`src/models.py`, `src/utils/stage.py`, `src/config.py` exactly as §6. Push to `main` and tell the group immediately.

**Verify:** `python -c "from src.models import Student; print(Student('007','A').index)"` prints `007` as a string.
**Commits:**
- `feat(core): add shared dataclasses for students, cells and attendance`
- `feat(core): add abstract Stage base class for pipeline steps`
- `feat(core): add central config with paths and shared constants`

---

### T3 — Logging and timing
`src/utils/logging.py` → `get_logger(name)`, format `[HH:MM:SS] LEVEL  module | message`, level controlled by a `DEBUG` flag.
`src/utils/timing.py` → `@timed` decorator recording seconds into a module-level dict for later reporting, and logging `stage 'binarize' finished in 0.42 s`.

**Verify:** a throwaway script logs a line and a duration.
**Commits:**
- `feat(utils): add project logger with consistent formatting`
- `feat(utils): add timed decorator to measure stage duration`

---

### T4 — Bootstrap fixtures  *(blocking — unblocks M4, M5, M6, M7, M8)*
`tools/make_fixtures.py`. Deliberately crude one-liner OpenCV, **not** the real pipeline. Takes the first sheet and writes:

```
data/fixtures/warped.png        # manual crop, no perspective correction
data/fixtures/grey.png
data/fixtures/binary.png        # Otsu, inverted so ink is white
data/fixtures/cell_sample.png   # one hand-picked signature cell, BGR
```

Docstring must state clearly: *temporary scaffolding so downstream modules can start; delete once M2–M4 land.*

**Verify:** the four files exist and open correctly.
**Commit:** `chore(tools): generate bootstrap fixtures so downstream modules can start`

---

### T5 — Progress viewer
`src/viz/progress.py`:

```python
class ProgressViewer:
    def __init__(self, sheet_date: str, show: bool = True, save: bool = True): ...
    def add(self, name: str, image: np.ndarray, cmap: str | None = None) -> None: ...
    def save_all(self) -> None:          # outputs/steps/<date>/NN_name.png
    def show_montage(self) -> None:      # matplotlib grid, titled, tight_layout
```

Rules: preserve insertion order; auto-number from 01; convert BGR→RGB for Matplotlib; handle single-channel images with a grey colormap; log `[3/8] binarize … saved` as each is added; the montage grid shape adapts to the number of steps.

**Verify:** feed it 4 test images → `outputs/steps/test/01_*.png` … `04_*.png` exist and a montage window opens.
**Commits:**
- `feat(viz): add ProgressViewer to collect pipeline step images`
- `feat(viz): save numbered step images to outputs/steps`
- `feat(viz): add matplotlib montage of all processing steps`
- `fix(viz): convert bgr to rgb and handle single channel images`

---

### T6 — Stubs
`src/stubs.py` per §8.

**Verify:** each stub instantiates and `run({})` returns a dict with the right keys.
**Commit:** `feat(core): add placeholder stages so the pipeline runs end to end`

---

### T7 — Pipeline runner
`src/pipeline.py`:

```python
class Pipeline:
    def __init__(self, stages: list[Stage], viewer: ProgressViewer | None = None): ...
    def run(self, sheet: SheetMeta, students: list[Student]) -> dict: ...
```

Behaviour: build `ctx`; for each stage — log start, time it, call `run`, push `figures()` into the viewer, log finish. On exception: log `stage 'table' failed: <message>` and re-raise wrapped in a `PipelineError` carrying the stage name. Never continue past a failure.

**Verify:** a fake two-stage pipeline runs in order; a deliberately failing stage produces an error naming that stage.
**Commits:**
- `feat(pipeline): add Pipeline runner executing stages in fixed order`
- `feat(pipeline): forward stage figures to the progress viewer`
- `feat(pipeline): record per stage timings`
- `fix(pipeline): raise PipelineError naming the failing stage`

---

### T8 — `sams.py`
Per §7. Assemble the stage list from real modules where available, stubs otherwise, chosen by a single `STAGES` list at the top of the file so swapping one stage is a one-line change.

**Verify:** `python sams.py data/sheets/<first>.png data/info.xml` runs to the summary table using stubs, and `python sams.py missing.png data/info.xml` exits with code 2 and a friendly message.
**Commits:**
- `feat(cli): add sams.py entry point with argument parsing and validation`
- `feat(cli): wire stage list into the sams pipeline`
- `feat(cli): print attendance summary table after processing`
- `fix(cli): exit with a clear message when input files are missing`

---

### T9 — `infovis.py` and `investigate.py`
Thin wrappers only. Argument parsing, validation, friendly errors, then delegate. While M8 and M9 are unfinished, print `module not ready yet — this command will work once M9/M8 lands` and exit 0.

**Verify:** both commands run and produce the placeholder message; `--help` works on all three.
**Commits:**
- `feat(cli): add infovis.py entry point for attendance charts`
- `feat(cli): add investigate.py entry point for signature checking`
- `fix(cli): handle unknown student index with helpful output`

---

### T10 — Tests
`tests/test_pipeline.py` per §11.

**Verify:** `pytest -q` passes.
**Commit:** `test(pipeline): cover stage ordering, error reporting and viewer output`

---

### T11 — Integration (one commit per real module swapped in)
As each member merges, delete their stub from `src/stubs.py`, wire the real stage into `STAGES`, and run the full command on all sheets.

**Verify after each swap:** `python sams.py <sheet> data/info.xml` still reaches the summary table.
**Commit pattern:** `refactor(pipeline): replace <stage> stub with real implementation`

---

### T12 — Freeze and package
Fill `README.md`: what it is, install, the three commands, folder map, known limits. Pin versions in `requirements.txt` from `pip freeze`. Produce the three figures in §13 with `tools/make_m1_figures.py`. Confirm `grep -r "STUB" src/` returns nothing. Tag.

**The `v1.0` tag is the group's, not M1's.** `src/stubs.py` cannot be empty until T11 has swapped in all seven real modules, so the STUB check cannot pass while M1 is the only work on `main`. M1 finishing tags `v0.1-m1-core` — everything M1 owns is done, the three commands run, the stubs are still standing in. `v1.0` waits for the last swap.

**Verify:** a clean clone into a fresh venv runs all three commands.
**Commits:**
- `docs(readme): add setup, usage and folder guide`
- `chore(release): pin dependency versions`
- `chore(release): tag v1.0 prototype submission`

---

## 9.2 M2 — Acquisition & Geometry

**Branch** `feat/m2-geometry` · **Owns** `src/io/image_loader.py`, `src/preprocess/deskew.py`, `tests/test_geometry.py`
**Reads** `ctx["sheet"]` · **Writes** `ctx["bgr"]`, `ctx["warped"]` · **Blocks** M3 and M5

`ctx["warped"]` is the single most important output in the project. Everything downstream reads it.

### What T0 measured that changes your job

- All five photos are **3024 x 4032, upright, and carry no EXIF orientation tag**. Keep the EXIF rotation code — it is one line and phone photos usually do have it — but do not expect it to fire here, and do not let its absence be treated as an error.
- The paper is photographed on a **pale desk** in every shot. Paper-against-cream is low contrast, so Canny plus largest-contour will fail on some sheets. **The fallback path in T4 is not optional garnish; budget real time for it.**
- The sheet carries **two tables** — a one-row lecture header table above the student table. Your crop must keep **both**, and must not cut the right hand `Signature` column. M5 decides which table is which; you must not make that decision for them by cropping one away.

### Contract

```python
# src/io/image_loader.py
def load_image(path: str | Path) -> np.ndarray:
    """BGR uint8, EXIF rotation applied. FileNotFoundError if missing,
    ValueError if not a decodable image. Both with clear messages."""

def resize_to_width(bgr: np.ndarray, width: int = 1600) -> np.ndarray:
    """Downscale only, never upscale, aspect ratio preserved."""

# src/preprocess/deskew.py
class GeometryStage(Stage):
    name = "geometry"
    def run(self, ctx: dict) -> dict: ...

def find_sheet_corners(bgr) -> np.ndarray | None:
    """(4, 2) float32, ordered top-left, top-right, bottom-right, bottom-left.
    None when not found — that is a normal outcome, not an error."""

def four_point_warp(bgr, corners) -> np.ndarray: ...
def estimate_skew_angle(grey) -> float: ...
```

`src/config.py` under `# --- M2 geometry ---`:

```python
TARGET_WIDTH = 1600
CANNY_LOW, CANNY_HIGH = 50, 150
MIN_SHEET_AREA_RATIO = 0.30
MAX_SKEW_CORRECTION_DEG = 15.0
BORDER_TRIM_PX = 6
```

### Tasks

**T1 — Loader.** `cv2.imread`, then EXIF orientation via Pillow. Clear errors. Add `resize_to_width` and downscale to `TARGET_WIDTH` — 3024 px wide is four times more than any later stage needs and makes every run slow.
- `feat(io): add image loader with validation and clear errors`
- `fix(io): correct EXIF orientation on phone photos`
- `feat(io): add aspect-preserving downscale helper`

**T2 — Corners, contour method.** Grey → Gaussian blur → Canny → `findContours` → largest → `approxPolyDP` with a tolerance loop until 4 points → order by sum and difference. Reject below `MIN_SHEET_AREA_RATIO`.
- `feat(preprocess): detect sheet outline with canny and contours`
- `feat(preprocess): order corner points consistently top-left first`
- `fix(preprocess): reject contours smaller than minimum area ratio`

**T3 — Perspective warp.** `getPerspectiveTransform` + `warpPerspective`. Output size from the longest opposite edges so the sheet is not squashed.
- `feat(preprocess): add four point perspective warp to top-down view`

**T4 — Fallback (expect to use it).** No corners → do not crash. Whole image plus a rotation correction from `estimate_skew_angle`: `HoughLinesP`, median angle of near-horizontal lines, clamped to `MAX_SKEW_CORRECTION_DEG`. The table borders are the strongest straight lines on the page, so measure the skew from those rather than the paper edge.
- `feat(preprocess): add hough based skew angle estimation`
- `feat(preprocess): fall back to rotation only when corners not detected`
- `feat(preprocess): log which geometry path was used`

**T5 — Border trim.** Shave `BORDER_TRIM_PX` off each edge after warping so leftover desk does not become a table line for M5.
- `feat(preprocess): trim residual border after warping`

**T6 — Wrap as a Stage.** `figures()` returns original, edge map, corner overlay, warped.
- `feat(preprocess): wrap geometry logic in GeometryStage class`

**T7 — All five sheets.** Record which path each used and whether the result is straight. **Verify by eye that both tables and all five columns survive the crop** — that is the acceptance test M5 will hold you to. Tune Canny in `config.py` only.
- `fix(preprocess): tune canny thresholds for low contrast desk backgrounds`
- `docs(preprocess): note per-sheet geometry results`

**Verify:** `python sams.py data/sheets/<any>.png data/info.xml --no-show` reaches the summary, and `outputs/steps/<date>/02_warped.png` shows a flat sheet with the full student table.

**Figures:** `m2_original_vs_warped.png`, `m2_corner_detection.png`, `m2_warp_steps.png`, `m2_skew_correction.png`, `m2_all_sheets_grid.png`

---

## 9.3 M3 — Greyscale & Enhancement

**Branch** `feat/m3-enhance` · **Owns** `src/preprocess/enhance.py`, `tests/test_enhance.py`
**Reads** `ctx["warped"]` · **Writes** `ctx["grey"]` · **Blocks** M4

### What T0 measured that changes your job

- All five sheets are **real colour** — no greyscale sheet to special-case.
- The photos have visible **paper texture, fold creases and a soft shadow gradient** across the page. Shadow removal (T3) is the task that actually matters here; the greyscale method barely moves the result. Spend your time accordingly, and say so honestly in the report.
- Your output feeds M4, whose output feeds M5's **line detection**. An enhancement that looks lovely but thins the printed table lines is a bad enhancement. Check with M5, not just with your eyes.

### Contract

```python
class EnhanceStage(Stage):
    name = "enhance"
    def run(self, ctx: dict) -> dict: ...

def to_grey(bgr, method: str = "luminosity") -> np.ndarray:
    """'average' | 'luminosity' | 'lightness' | 'max_channel'.
    Write average and luminosity yourself in NumPy, not cv2.cvtColor,
    so the report can explain 0.299 R + 0.587 G + 0.114 B."""

def denoise(grey, method: str = "bilateral") -> np.ndarray: ...
def remove_shadow(grey) -> np.ndarray: ...
def enhance_contrast(grey, method: str = "clahe") -> np.ndarray: ...
```

Unknown method name raises `ValueError` — never fall through silently to a default.

`src/config.py` under `# --- M3 enhancement ---`:

```python
GREY_METHOD = "luminosity"
DENOISE_METHOD = "bilateral"
BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE = 9, 75, 75
MEDIAN_KSIZE = 3
SHADOW_KERNEL = 25
CLAHE_CLIP, CLAHE_GRID = 2.0, (8, 8)
```

### Tasks

**T1 — Greyscale, four ways.** `average` and `luminosity` in plain NumPy so you can explain why green dominates (eye sensitivity). Compare all four on one sheet.
- `feat(preprocess): add average and luminosity greyscale with numpy`
- `feat(preprocess): add lightness and max-channel greyscale variants`
- `docs(preprocess): note why luminosity weights differ per channel`

**T2 — Denoise, four ways.** Gaussian, median, bilateral, non-local means. **Measure** PSNR/SSIM and runtime — do not claim bilateral wins, show it.
- `feat(preprocess): add gaussian and median denoising`
- `feat(preprocess): add bilateral filter to keep stroke edges sharp`
- `feat(preprocess): add non-local means denoising option`
- `test(preprocess): measure psnr and runtime for each denoise method`

**T3 — Shadow removal (the big win).** Dilate with a large kernel then median blur to estimate the lighting map, divide, rescale to 0–255.
- `feat(preprocess): estimate background lighting with morphology`
- `feat(preprocess): divide by background to flatten uneven lighting`
- `fix(preprocess): rescale to full range after shadow division`

**T4 — Contrast.** Global histogram equalisation vs CLAHE, with the histograms shown.
- `feat(preprocess): add global histogram equalisation`
- `feat(preprocess): add clahe local contrast enhancement`

**T5 — Wrap as a Stage.** Chain grey → shadow → denoise → contrast, every step switchable from `config.py`. `figures()` returns each intermediate.
- `feat(preprocess): wrap enhancement chain in EnhanceStage class`

**T6 — Tune with M4 and M5.** Your best-looking image is not always the one that binarises best, and the one that binarises best is not always the one whose table lines survive. Agree final settings with both and write them into `config.py`.
- `fix(preprocess): tune clahe clip limit after binarisation feedback`
- `docs(preprocess): record agreed enhancement settings`

**Verify:** `outputs/steps/<date>/03_grey.png` on the sheet with the worst shadow is evenly lit corner to corner.

**Figures:** `m3_greyscale_methods.png`, `m3_histograms.png`, `m3_denoise_comparison.png`, `m3_denoise_metrics.png`, `m3_shadow_removal.png`

---

## 9.4 M4 — Binarisation & Morphology

**Branch** `feat/m4-binarize` · **Owns** `src/preprocess/binarize.py`, `tests/test_binarize.py`
**Reads** `ctx["grey"]` · **Writes** `ctx["binary"]` · **Blocks** M5, and later helps M8

**Ink = 255 (white), paper = 0 (black). This never changes.** `THRESH_BINARY_INV` gives it to you. Everyone downstream assumes it, and getting the polarity backwards silently breaks M5, M6 and M7 at once.

### What T0 measured that changes your job

- Your real customer is **M5's line detection**, not the human eye. The printed table lines are thin — closing that repairs a broken pen stroke can also weld a signature to the border line, and M6 then measures a signature that is 40% table. Tune with M5 watching.
- Two cells on `05.07.2019` and `21.06.2019` contain **faint marks that are not signatures**. Do not tune your threshold until they disappear — losing them makes M7's job look easy and the accuracy numbers dishonest. Keep them, and let M7's rule reject them.

### Contract

```python
class BinarizeStage(Stage):
    name = "binarize"
    def run(self, ctx: dict) -> dict: ...

def threshold_global(grey, value: int = 127) -> np.ndarray: ...
def threshold_otsu(grey) -> tuple[np.ndarray, int]:
    """Binary image AND the chosen threshold. Search written by hand — see T2."""
def threshold_adaptive(grey, method="gaussian", block=35, c=10) -> np.ndarray: ...
def threshold_sauvola(grey, window=25) -> np.ndarray: ...
def morph_clean(binary, open_k=2, close_k=3) -> np.ndarray: ...
def skeletonize_ink(binary) -> np.ndarray: ...
def clean_signature_crop(mask) -> np.ndarray:
    """Small-crop variant for M8. Agree what they need before writing it."""
```

`src/config.py` under `# --- M4 binarisation ---`:

```python
BINARIZE_METHOD = "adaptive"      # global | otsu | adaptive | sauvola
ADAPTIVE_BLOCK, ADAPTIVE_C = 35, 10
SAUVOLA_WINDOW = 25
MORPH_OPEN_K, MORPH_CLOSE_K = 2, 3
```

### Tasks

**T1 — Global baseline, kept as a failure exhibit.** One fixed value cannot suit a photo with a shadowed corner. Keep the failure image, it is a good figure.
- `feat(preprocess): add global fixed threshold baseline`

**T2 — Otsu, written by hand.** Do not stop at `cv2.THRESH_OTSU`. Implement it: 256-bin normalised histogram; for each candidate `t` compute class weights `w0, w1` and means `m0, m1`; maximise between-class variance `w0 * w1 * (m0 - m1)²`. Then check your value matches OpenCV within ±1 — that check is both a strong test and a strong report point.
- `feat(preprocess): implement otsu threshold search with numpy`
- `feat(preprocess): return chosen threshold value alongside binary image`
- `test(preprocess): verify custom otsu matches opencv within one level`

**T3 — Adaptive.** Mean and Gaussian. Normally the winner on phone photos. Sweep `block` and `c`, record the best pair. Force odd block sizes or raise a clear `ValueError`.
- `feat(preprocess): add adaptive mean and gaussian thresholding`
- `fix(preprocess): force odd block size and validate parameters`
- `docs(preprocess): record adaptive block and c sweep results`

**T4 — Sauvola.** `skimage.filters.threshold_sauvola`, built for documents. Compare against adaptive.
- `feat(preprocess): add sauvola local thresholding for documents`

**T5 — Compare properly (this earns marks).** Per method, per sheet: ink pixel percentage (a few percent, not 40), connected component count, **whether M5's table lines survive**, runtime.
- `feat(preprocess): add binarisation comparison harness with metrics`
- `docs(preprocess): record method comparison across all five sheets`

**T6 — Morphology.** Opening kills specks, closing repairs broken strokes. Try `MORPH_ELLIPSE` and `MORPH_RECT`.
- `feat(preprocess): add morphological opening to remove speckle noise`
- `feat(preprocess): add closing to repair broken pen strokes`
- `fix(preprocess): reduce closing kernel to stop strokes merging with table lines`

**T7 — Skeletonisation.** `skimage.morphology.skeletonize`. M6 needs `stroke_length` and M8 needs it too.
- `feat(preprocess): add skeletonisation for one pixel wide strokes`

**T8 — Wrap as a Stage.** Method from `config.py`, `figures()` returns raw and cleaned.
- `feat(preprocess): wrap thresholding chain in BinarizeStage class`

**T9 — Help M8.** `clean_signature_crop(mask)` tuned for small crops, not whole sheets. Ask M8 what they need first.
- `feat(preprocess): add signature crop cleaning helper for recognition`

**Verify:** `set(np.unique(ctx["binary"])) <= {0, 255}`, ink is white, and M5 confirms the table lines are unbroken.

**Figures:** `m4_threshold_comparison.png`, `m4_otsu_histogram.png` (**your best figure — the between-class variance curve with the chosen threshold marked**), `m4_global_failure.png`, `m4_morphology.png`, `m4_metrics.png`

---

## 9.5 M5 — Table Detection

**Branch** `feat/m5-table` · **Owns** `src/table/line_detect.py`, `src/table/grid_builder.py`, `src/table/cell_extract.py`, `tests/test_table.py`
**Reads** `ctx["binary"]`, `ctx["warped"]` · **Writes** `ctx["grid"]`, `ctx["cells"]` · **Blocks** M6, and so M7, M8, M9

The hardest single module. Get a rough `list[Cell]` into M6's hands early — rough and early beats perfect and late.

### What T0 measured that changes your job — read this twice

**Your brief document is wrong on all four of these. This file wins.**

1. **There are 5 columns, not 4.** `No | Student No | Title | Student Name | Signature`. So `EXPECTED_COLS = 5` and the signature is column **4**. Always use `config.SIGNATURE_COL`, never a literal.
2. **There are two tables on the page.** A one-row lecture header table (`Date | time | Lecture's Name | Signatue`) sits directly above the student table, and **it also has a signature in its last column — the lecturer's.** If you pick the wrong table, or merge the two, the lecturer's signature is reported as a student's and the whole system is wrong in a way that still looks plausible. Select the **lower** block of horizontal lines, the one with 7 lines bounding 6 data rows plus a header. Log which one you took.
3. **6 data rows on every sheet, one header row.** `Grid.header_rows = 1`, `row = 0` is student 1. Row count is a hard check: if you do not get 6, warn loudly (§10) rather than silently returning 5.
4. **Signatures cross cell borders.** On `31.05.2019` and `05.07.2019` a signature runs well into the row below. Crop with a small vertical pad rather than a hard cut, and see §14 decision 1 — you and M6 agree the rule together.

### Contract

```python
# src/table/line_detect.py
def detect_horizontal_lines(binary, min_len_ratio: float = 0.5) -> list[int]: ...
def detect_vertical_lines(binary, min_len_ratio: float = 0.5) -> list[int]: ...
def line_mask(binary, orientation: str) -> np.ndarray: ...

# src/table/grid_builder.py
@dataclass
class Grid:
    xs: list[int]              # vertical line positions, left to right
    ys: list[int]              # horizontal line positions, top to bottom
    header_rows: int = 1
    @property
    def n_rows(self) -> int: ...
    @property
    def n_cols(self) -> int: ...
    def cell_bbox(self, row: int, col: int) -> tuple[int, int, int, int]: ...
    def cells(self, col: int | None = None) -> list[Cell]: ...

def build_grid(xs: list[int], ys: list[int]) -> Grid: ...
def select_student_table(row_bands: list[list[int]]) -> list[int]:
    """Given the horizontal line groups on the page, return the ones
    belonging to the student table. The lecture header table is discarded."""

# src/table/cell_extract.py
class TableStage(Stage):
    name = "table"
    def run(self, ctx: dict) -> dict: ...

def crop_cell(warped, bbox, inset: int = 4, pad_y: int = 0) -> np.ndarray:
    """Inset excludes the border; pad_y keeps an overflowing signature."""
```

`ctx["cells"]` is the signature column only, `.image` cropped from **`warped` in colour** — never the binary image, M6 needs the pen colour — with `.row` set and no gaps.

`src/config.py` under `# --- M5 table detection ---`:

```python
H_KERNEL_RATIO = 0.30
V_KERNEL_RATIO = 0.30
LINE_MERGE_TOL = 8
MIN_ROW_HEIGHT = 18
MIN_COL_WIDTH = 25
CELL_INSET = 4
CELL_PAD_Y = 6            # keeps signatures that overflow the row
EXPECTED_COLS = 5         # No, Student No, Title, Student Name, Signature
EXPECTED_DATA_ROWS = 6
```

### Tasks

**T1 — Line masks by morphology.** Horizontal: erode then dilate with a `(width * H_KERNEL_RATIO, 1)` kernel; vertical with `(1, height * V_KERNEL_RATIO)`. Far more reliable than Hough on printed tables — do this first.
- `feat(table): extract horizontal line mask with wide morphology kernel`
- `feat(table): extract vertical line mask with tall morphology kernel`
- `feat(table): combine masks to visualise the detected table skeleton`

**T2 — Positions from projection profiles.** Sum the mask along an axis, find peaks, merge peaks closer than `LINE_MERGE_TOL`.
- `feat(table): convert line masks to positions using projection profiles`
- `feat(table): merge nearby peaks into single line positions`

**T3 — Pick the student table.** Group the horizontal lines into bands separated by large gaps. The lecture header table is a short band of 2–3 lines near the top; the student table is the tall band of 7. Take the student table, log the choice, and keep both drawn in a figure so the report can show the trap.
- `feat(table): group horizontal lines into table bands`
- `feat(table): select the student table and discard the lecture header table`
- `fix(table): warn when the expected two table bands are not found`

**T4 — Hough as cross-check.** `HoughLinesP`, keep lines within ±5° of axis-aligned. Compare with T2. A cross-check and a figure — not the primary method.
- `feat(table): add hough line detection as cross check`
- `docs(table): compare morphology and hough line positions`

**T5 — Grid repair.** Real photos lose lines. Find the median row spacing, insert a line where a gap is close to a multiple of it, drop lines below `MIN_ROW_HEIGHT` / `MIN_COL_WIDTH`, and warn loudly when the column count is not `EXPECTED_COLS` or the data row count is not `EXPECTED_DATA_ROWS`.
- `feat(table): estimate median row spacing`
- `feat(table): insert missing horizontal lines from regular spacing`
- `fix(table): drop duplicate lines below minimum spacing`
- `feat(table): warn when detected row or column count is unexpected`

**T6 — Header handling.** Set `Grid.header_rows = 1` so data row 0 is the first student.
- `feat(table): detect and skip the header row`

**T7 — Cell cropping.** `CELL_INSET` to keep the border out, `CELL_PAD_Y` to keep an overflowing signature in. Build `Cell` objects for `config.SIGNATURE_COL` from the **warped colour image**.
- `feat(table): crop cells with inset to exclude table borders`
- `feat(table): pad cell crops vertically for overflowing signatures`
- `feat(table): build Cell objects for the signature column`
- `fix(table): crop from warped colour image so pen colour is preserved`

**T8 — Wrap as a Stage.** `figures()` returns the line masks and a grid overlay on the warped sheet.
- `feat(table): wrap grid detection in TableStage class`

**T9 — All five sheets.** Record detected rows and columns per sheet against the true 6 and 5. Tune ratios in `config.py` only.
- `fix(table): tune kernel ratios for sheets with faint printed lines`
- `docs(table): record per sheet row and column detection accuracy`

**Verify:** on all five sheets, `len(ctx["cells"]) == 6` and every crop shown in `m5_cells_numbered.png` is a signature box and not a name.

**Figures:** `m5_line_masks.png`, `m5_projection_profiles.png`, `m5_grid_overlay.png`, `m5_cells_numbered.png`, `m5_grid_repair.png`, `m5_two_tables.png` (**the header table and the student table distinguished — this is the figure that shows you understood the page**), `m5_hough_vs_morphology.png`

---

## 9.6 M6 — Ink Segmentation

**Branch** `feat/m6-ink` · **Owns** `src/detect/cell_clean.py`, `src/detect/ink_mask.py`, `tests/test_ink.py`
**Reads** `ctx["cells"]` · **Writes** `ctx["ink"]` · **Blocks** M7, and supplies M8's crops

The brief says students sign **using different colour pens**. Handling that is your headline contribution.

### What T0 measured that changes your job

- The five sheets are signed in **blue ballpoint almost throughout**, with **one red mark** on `05.07.2019`. So your multi-colour machinery is right, but you cannot prove it on this data alone — **make synthetic green and black test cells** and prove it there, then say plainly in the report that the sample happened to be mostly blue. A measured limitation beats an unproven claim.
- **Two cells contain ink that is not a signature**: the lecturer's handwritten `ab` on `21.06.2019`, and a small stray red tick on `05.07.2019`. Both mean *absent*. **Your job is not to decide that** — it is to supply features sharp enough that M7 can. `stroke_length`, `filled_ratio`, `aspect` and `components` are what separate a signature from a two-letter word or a 3 mm tick. See §14 decision 2.
- **Signatures overflow their cells.** A stroke starting in row 3 can end in row 4. Attribute a connected component to the row holding most of its pixels, and agree the rule with M5 (§14 decision 1).
- **Crop filenames: use the student index, not the row number.** Your brief says `row_<n>.png`. It must be `outputs/cells/<sheet_date>/<index>.png` and `<index>_mask.png`, because that is what §5.4 defines, what `investigate.py` globs to count samples, and what M8 looks up. The `Cell` does not know its index — M7 attaches it — so either write the crops in `DecisionStage` after mapping, or have M5 pass the row and M7 rename. **Agree this with M7 before you write it.**

### Contract

```python
# src/detect/cell_clean.py
def remove_table_lines(cell_bgr) -> np.ndarray: ...
def trim_to_content(mask, pad: int = 4) -> tuple[np.ndarray, tuple]: ...

# src/detect/ink_mask.py
class InkStage(Stage):
    name = "ink"
    def run(self, ctx: dict) -> dict: ...

def ink_mask(cell_bgr, method: str = "combined") -> np.ndarray:
    """'hsv' | 'lab' | 'saturation' | 'darkness' | 'combined'. ink = 255."""

def dominant_pen_colour(cell_bgr, mask) -> str:
    """'blue' | 'black' | 'red' | 'green' | 'other'."""

def ink_features(mask) -> dict:
    """ink_ratio, components, stroke_bbox, aspect, stroke_length,
    filled_ratio, centroid_offset. Key names agreed with M7 first."""
```

`src/config.py` under `# --- M6 ink segmentation ---`:

```python
INK_METHOD = "combined"
SAT_MIN = 60
VAL_MAX = 200
DARK_MAX = 160
MIN_BLOB_AREA = 12
CELL_PAD = 4
PEN_HUE_RANGES = {
    "blue":  [(90, 130)],
    "green": [(40, 85)],
    "red":   [(0, 10), (170, 180)],
}
```

### Tasks

**T1 — Clean the crop.** Remove long straight runs touching the crop edge, drop blobs touching the outer 2-pixel frame. M5's inset helps but does not finish the job.
- `feat(detect): remove leftover table border lines inside cell crops`
- `fix(detect): drop blobs touching the crop edge`

**T2 — Colour ink mask (headline work).** HSV. Coloured pens: `saturation >= SAT_MIN` — white paper has almost none. Black pen has no saturation either, so add `value <= VAL_MAX`. Combine with OR. Also try LAB's `a`/`b` channels and report which won.
- `feat(detect): add hsv saturation mask for coloured pen ink`
- `feat(detect): add value threshold branch to catch black pen`
- `feat(detect): add lab colour space ink mask variant`
- `feat(detect): combine colour and darkness masks into one ink mask`
- `docs(detect): compare hsv and lab masking results`

**T3 — Clean the mask.** Drop components under `MIN_BLOB_AREA`, small close to join broken strokes. **Never dilate heavily** — it inflates `ink_ratio` and makes empty cells look signed.
- `feat(detect): drop connected components below minimum area`
- `feat(detect): close small gaps within pen strokes`

**T4 — Pen colour.** Mean hue of ink pixels through `PEN_HUE_RANGES`; low saturation means black. Handle red's wrap-around at 0/180.
- `feat(detect): identify dominant pen colour from ink hue`
- `feat(detect): count pen colour usage per sheet`

**T5 — Features for M7.** Full `InkResult`. `stroke_length` comes from M4's `skeletonize_ink` — ask, do not rewrite it. Agree every key name with M7 **before** writing.
- `feat(detect): compute ink ratio and connected component count`
- `feat(detect): add stroke bounding box and aspect ratio features`
- `feat(detect): add skeleton length and fill ratio features`

**T6 — Save crops for M8.** Colour crop and mask to `outputs/cells/<sheet_date>/<index>.png` and `<index>_mask.png`, paths into `InkResult.crop_path` / `.mask_path`. Without these there is no `investigate.py`.
- `feat(detect): save cell crops and masks for signature recognition`

**T7 — Wrap as a Stage.** `figures()` returns a montage of the cells and their masks.
- `feat(detect): wrap ink segmentation in InkStage class`

**T8 — Tune on all five sheets.** Look at failures with your own eyes: faint ink, a signature crossing rows, a printed dot read as ink.
- `fix(detect): lower saturation threshold for faded ink`
- `docs(detect): record ink segmentation failure cases`

**Verify:** the **three** genuinely empty cells — `28.06.2019 / 10009301`, `28.06.2019 / 10009302`, `05.07.2019 / 10009301` — give near-zero `ink_ratio`, and every real signature gives a clearly higher one. The other two absences (`21.06.2019 / 10009306` and `05.07.2019 / 10009303`) **do** contain ink and must not be masked away; they are M7's problem, not yours.

**Figures:** `m6_colour_spaces.png`, `m6_hue_scatter.png`, `m6_mask_panels.png`, `m6_border_removal.png`, `m6_pen_colour_counts.png`, `m6_empty_vs_signed.png`, `m6_ink_that_is_not_a_signature.png` (**the `ab` cell and the red tick beside a real signature, with all their features printed — this is the figure the discussion section needs**)

---

## 9.7 M7 — Decision & Database

**Branch** `feat/m7-decision-db` · **Owns** `src/io/xml_parser.py`, `src/io/db.py`, `src/detect/presence.py`, `tools/seed_db.py`, `tests/test_decision.py`, `tests/test_db.py`
**Reads** `ctx["ink"]`, `ctx["students"]` · **Writes** `ctx["records"]`, the database · **Blocks** M8 and M9

Last stage of the pipeline and the only source of data for M8 and M9. **Push the database schema and `tools/seed_db.py` before you write a line of decision logic** — two people are idle until you do.

### What T0 measured that changes your job

1. **`info.xml` is not the shape your brief guesses.** It is not `<info><students><student>`. The real file, reconstructed and committed, is:
   `nsbm/students/batches/batch/student` with `index`, `title`, `name`, plus a sibling `nsbm/subject` holding `code`, `name`, `degree`, `lecturer`.
   **Find students with `.//student`** and read the subject with `.//subject`. That survives the batch element changing shape, which matters because the brief's own Figure 1 shows `<15>` — a tag that starts with a digit and therefore parses in nothing. See §4 deviation 4.
2. **Indices are 8 digits**, e.g. `10000409`. Still strings. `int()` anywhere near an index is a bug.
3. **Positional mapping is safe here.** The XML was transcribed from the sheets in row order, so sheet row *n* is XML student *n*. Still warn when the counts disagree (§10). OCR verification of the printed index column stays optional.
4. **`data/ground_truth.csv` already exists** — 30 rows, hand-transcribed by M1 in T0, with a `note` column naming the two awkward cells. Your T5 is to *verify and use* it, not create it. Check a sample against the images yourself before you trust it.
5. **The interesting accuracy cases are known in advance.** `21.06.2019 / 10009306` holds `ab`; `05.07.2019 / 10009303` holds a stray red tick. Both are ink and both are absent. Report your accuracy with and without them, and say which way you leaned on false-positive against false-negative.
6. **`src/io/db.py` must expose `known_indices()` and `is_empty()`** as module-level functions — §6.5. `infovis.py` and `investigate.py` call them to give helpful errors. Wrap the `Database` class; do not make the CLIs construct one.

### Contract

```python
# src/io/xml_parser.py
def parse_info(path) -> tuple[list[Student], dict]:
    """Students in sheet order, plus meta: subject_code, subject_name, lecturer."""

def parse_students(path) -> list[Student]:
    """Thin wrapper — this is what sams.py imports (§6.5)."""

# src/detect/presence.py
class DecisionStage(Stage):
    name = "decision"
    def run(self, ctx: dict) -> dict: ...

def decide(result: InkResult) -> tuple[bool, float]: ...
def map_rows_to_students(cells, students) -> dict[int, str]: ...

# src/io/db.py
class Database:
    def __init__(self, path: Path = config.DB_PATH): ...
    def init_schema(self) -> None: ...
    def upsert_students(self, students) -> None: ...
    def upsert_sheet(self, meta, subject_code) -> int: ...
    def save_attendance(self, records, sheet_id) -> None: ...
    def save_signature(self, student_index, sheet_id, **fields) -> None: ...
    def get_attendance(self, student_index) -> list[dict]: ...
    def get_signatures(self, student_index) -> list[dict]: ...
    def get_all_attendance(self) -> list[dict]: ...
    def get_student(self, student_index) -> dict | None: ...

def known_indices() -> list[str]: ...     # §6.5, called by the CLIs
def is_empty() -> bool: ...               # §6.5, called by the CLIs
```

Schema — four tables in `data/attendance.db`:

```sql
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
CREATE TABLE IF NOT EXISTS signatures (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    student_index TEXT NOT NULL,
    sheet_id      INTEGER NOT NULL,
    crop_path     TEXT,
    mask_path     TEXT,
    ink_ratio     REAL,
    components    INTEGER,
    aspect        REAL,
    stroke_length INTEGER,
    UNIQUE(student_index, sheet_id)
);
```

Parameterised queries everywhere (`?`), context-managed connections, `INSERT OR REPLACE` so re-running a sheet updates instead of duplicating.

`src/config.py` under `# --- M7 decision ---`:

```python
INK_RATIO_THRESHOLD = 0.012      # set from the sweep in T5, not by feel
MIN_COMPONENTS = 1
MIN_STROKE_LENGTH = 25
CONF_LOW, CONF_HIGH = 0.008, 0.030
```

### Tasks

**T1 — Database first.** Schema, `Database`, `init_schema`, the two module-level helpers, and `tools/seed_db.py` with fake data. Push and tell M8 and M9 the same hour.
- `feat(db): add sqlite schema for students, sheets and attendance`
- `feat(db): add Database class with context managed connections`
- `feat(db): add signatures table for recognition module`
- `feat(db): expose known_indices and is_empty for the cli helpers`
- `chore(tools): add db seeder with fake data for downstream development`

**T2 — XML parser.** `xml.etree.ElementTree`, `.//student`. Indices stay strings. Clear errors for malformed or missing tags.
- `feat(io): parse info.xml into student and subject records`
- `fix(io): keep student indices as strings to preserve leading zeros`
- `fix(io): give clear errors for malformed or missing xml tags`

**T3 — Row to student mapping.** Positional first. OCR verification of the printed index column (`pytesseract`, digits only) only if time allows; on disagreement, warn and trust the XML.
- `feat(detect): map sheet rows to students by position`
- `feat(detect): warn when row count and xml student count differ`
- `feat(detect): optional ocr verification of printed index column`

**T4 — The decision rule.** Never one number alone:
```
present = ink_ratio >= INK_RATIO_THRESHOLD
          and components >= MIN_COMPONENTS
          and stroke_length >= MIN_STROKE_LENGTH
```
Confidence scales `ink_ratio` between `CONF_LOW` and `CONF_HIGH`, reduced near the threshold. Borderline cells are flagged uncertain — being willing to say so is a strength.
- `feat(detect): add ink ratio threshold decision`
- `feat(detect): require minimum components and stroke length`
- `feat(detect): compute confidence score from ink ratio`
- `feat(detect): flag borderline cells as uncertain`

**T5 — Tune against ground truth.** Verify `data/ground_truth.csv` against the images, then sweep `INK_RATIO_THRESHOLD` and pick the best accuracy. **Show the sweep curve.** Report accuracy including and excluding the two ink-but-absent cells.
- `chore(data): verify hand labelled ground truth against the sheets`
- `feat(detect): add threshold sweep tool over ground truth`
- `fix(detect): set ink ratio threshold from sweep results`

**T6 — Wrap as a Stage and persist.** Decide → build records → upsert students, sheet, attendance and signatures in one transaction.
- `feat(detect): wrap decision logic in DecisionStage class`
- `feat(db): persist attendance and signature records per sheet`
- `fix(db): make re-processing a sheet update instead of duplicating rows`

**Verify:** all five sheets processed, 30 attendance rows and no duplicates on a second run; `sams.py` summary shows 6 students with the counts matching `ground_truth.csv`.

**Figures:** `m7_threshold_sweep.png`, `m7_ink_ratio_distribution.png` (**overlapping present/absent histograms with the threshold line — your main figure**), `m7_confusion_matrix.png`, `m7_accuracy_per_sheet.png`, `m7_er_diagram.png`

---

## 9.8 M8 — Signature Recognition

**Branch** `feat/m8-recognition` · **Owns** `src/recognise/preprocess_sig.py`, `src/recognise/features.py`, `src/recognise/matcher.py`, `src/recognise/report.py`, `tools/eval_recognition.py`, `tests/test_recognition.py`
**Reads** M7's `signatures` table and M6's saved crops · **Drives** `investigate.py`

The only part of the brief that explicitly promises higher grades: *"An attempt to distinguish the student signatures using any advanced library or self-created would result higher grades."* Treat it as the group's showcase. You are off the critical path, so you can work steadily — build against your own hand-cropped samples until M6 and M7 land.

### The problem you must solve first

Five sheets means **at most 5 samples per student and zero known forgeries**. You cannot train a model. The trick:

- Student A's signatures compared with each other → **genuine pairs**
- Student A's compared with B, C, D… → **impostor pairs**

Two score distributions, a cut-off where they separate, and real numbers: False Accept Rate, False Reject Rate, Equal Error Rate. That turns "we picked 0.62" into an experiment. **This is where the bonus marks are.**

### What T0 measured that changes your job

- **Six students, five sheets, and not everyone signed every sheet.** Counted from `data/ground_truth.csv`: two students have 5 samples, three have 4, one has 3 — **41 genuine pairs in total**. That is a small experiment. State the number in the report and do not present the EER as though it came from thousands of pairs.
- **`investigate.py` calls `matcher.investigate(index: str, save_only: bool) -> None`** — §6.5. Your brief shows `investigate(student_index, db)`; the CLI does not build a `Database`, so construct your own inside. Keep the brief's `InvestigationReport` if you like, but the entry point signature is fixed.
- **Crops are at `outputs/cells/<sheet_date>/<index>.png`** with `<index>_mask.png` beside them (§9.6 T6). `investigate.py` counts those files to decide whether there is anything to compare, and stops cleanly with exit 0 when there are fewer than two.
- **One student's signature genuinely varies a lot** — compare `10009306` across `31.05` and `12.07` by eye before you tune anything. If your threshold flags real variation as a mismatch, that is a finding, not a bug to hide.

### Contract

```python
# src/recognise/preprocess_sig.py
def normalise_signature(mask, size=(220, 120)) -> np.ndarray:
    """Trim to ink, keep aspect, pad, centre by centre of mass, resize."""

# src/recognise/features.py
def hog_vector(norm) -> np.ndarray: ...
def hu_moments(norm) -> np.ndarray: ...
def orb_keypoints(norm): ...
def shape_stats(norm) -> dict:
    """Your own: ink density, aspect, stroke length, Euler number,
    projection profiles, centre of mass, slant angle."""

# src/recognise/matcher.py
@dataclass
class MatchScore:
    ssim: float; hog: float; hu: float; orb: float
    custom: float; combined: float
    verdict: str              # "match" | "mismatch" | "uncertain"

def compare(a, b) -> MatchScore: ...
def investigate(index: str, save_only: bool = False) -> None:   # §6.5
    """Load every sample, compare all pairs, print the table, flag the
    outlier, save the figure. Never raises on too few samples."""
```

Every score comes out in 0–1 where 1 is identical. Distances need converting.

`src/config.py` under `# --- M8 recognition ---`:

```python
SIG_NORM_SIZE = (220, 120)
HOG_ORIENTATIONS, HOG_PPC, HOG_CPB = 9, (8, 8), (2, 2)
SCORE_WEIGHTS = {"ssim": 0.30, "hog": 0.30, "hu": 0.10, "orb": 0.15, "custom": 0.15}
MATCH_THRESHOLD = 0.62        # set from the EER experiment in T5, not guessed
UNCERTAIN_BAND = 0.08
USE_CNN_FEATURES = False
```

### Tasks

**T1 — Collect samples.** `Database.get_signatures(index)`, skipping absences. Fewer than 2 → clear message, no crash.
- `feat(recognise): load signature samples for a student from the database`
- `fix(recognise): handle students with fewer than two samples`

**T2 — Normalisation.** Nothing works without it: trim to ink → keep aspect → pad → centre by centre of mass → resize to `SIG_NORM_SIZE`. Use M4's `clean_signature_crop` rather than writing your own morphology.
- `feat(recognise): trim signature masks to the ink bounding box`
- `feat(recognise): centre signatures by centre of mass`
- `feat(recognise): resize to fixed box keeping aspect ratio`

**T3 — The five features.** SSIM (`skimage.metrics`), HOG + cosine (`skimage.feature`), Hu moments log-scaled (`cv2.HuMoments`), ORB + Lowe ratio test, and `shape_stats` — **your own, in plain NumPy**.
- `feat(recognise): add ssim similarity between normalised signatures`
- `feat(recognise): add hog descriptor with cosine similarity`
- `feat(recognise): add hu moment shape comparison`
- `feat(recognise): add orb keypoint matching with lowe ratio test`
- `feat(recognise): add custom shape statistics feature set`
- `feat(recognise): add projection profile correlation to custom score`
- `refactor(recognise): normalise all scores to a zero to one range`

**T4 — Combine.** Weighted sum from `SCORE_WEIGHTS`; above `MATCH_THRESHOLD` is a match, below by more than `UNCERTAIN_BAND` is a mismatch, between is uncertain.
- `feat(recognise): combine individual scores into one weighted score`
- `feat(recognise): add match, mismatch and uncertain verdicts`

**T5 — Genuine vs impostor (the bonus marks).** `tools/eval_recognition.py`: build both pair sets, plot both distributions on one axis, sweep the threshold, compute FAR/FRR, find the EER, set `MATCH_THRESHOLD` from it, and report which single feature separates best — that justifies your weights.
- `feat(eval): build genuine and impostor score pairs from all students`
- `feat(eval): plot score distributions for both pair types`
- `feat(eval): compute far, frr and equal error rate across thresholds`
- `fix(recognise): set match threshold from measured equal error rate`
- `docs(recognise): record which features separate genuine pairs best`

**T6 — Outlier inside one student.** Pairwise similarity matrix, mean similarity per sample, flag the lowest with its sheet date.
- `feat(recognise): build pairwise similarity matrix per student`
- `feat(recognise): flag the sample least similar to the rest`

**T7 — Output.** `python investigate.py 10000409` prints a pairwise table and a verdict:
```
Student 10000409 — M S Dilshanika Perera — 5 samples

  pair                      SSIM   HOG    HU     ORB    OWN    COMBINED
  31.05 vs 21.06            0.81   0.88   0.92   0.79   0.84     0.84  match
  31.05 vs 28.06            0.78   0.85   0.90   0.74   0.80     0.81  match
  21.06 vs 12.07            0.41   0.39   0.55   0.22   0.35     0.38  MISMATCH

Verdict: signature on 12.07.2019 does not match the others (mean 0.39 vs 0.82).
Flagged for manual review.
Figures saved to outputs/figures/m8_investigate_10000409.png
```
- `feat(recognise): print pairwise score table for investigate command`
- `feat(recognise): print final verdict with the flagged sheet date`

**T8 — Optional tier 3, only after T1–T7 work.** Pretrained ResNet18 embedding behind `USE_CNN_FEATURES`. **No training, and no Siamese network** — not enough data, not enough time. If it is slow or unhelpful, switch it off and report that honestly. A measured negative result still earns marks.
- `feat(recognise): add optional pretrained cnn embedding features`
- `docs(recognise): report cnn embedding results and runtime cost`

**Verify:** `python investigate.py 10000409` prints a table and a verdict for all 10 of its pairs; `compare(x, x)` scores near 1.0. Every student in this data has at least 3 samples, so the fewer-than-two path has no real case here — cover it with a unit test, not by hoping.

**Figures:** `m8_normalisation_steps.png`, `m8_score_distributions.png` (**genuine vs impostor with the threshold line — your headline figure**), `m8_far_frr_curve.png`, `m8_similarity_matrix.png`, `m8_orb_matches.png`, `m8_feature_comparison.png`, `m8_flagged_example.png`

---

## 9.9 M9 — Visualisation & QA

**Branch** `feat/m9-viz-qa` · **Owns** `src/viz/charts.py`, `src/viz/style.py`, `tools/run_all_sheets.py`, `tools/make_report_assets.py`, `tests/test_charts.py`, `tests/test_e2e.py`
**Reads** M7's database · **Drives** `infovis.py` · **Also** the group's QA

Data visualisation is one of the module's **two named key technologies**. This module is judged on more than "it drew a bar chart". `src/viz/progress.py` is M1's — do not touch it.

### What T0 measured that changes your job

- **6 students × 5 sheets = 30 cells.** Your heat map is 6 rows by 5 columns. That is small, which is a gift: every cell can carry a label, and the whole class fits on one readable figure. Design for that rather than for a 200-student grid.
- **Dates must sort chronologically, not as strings.** `05.07.2019` sorts before `21.06.2019` alphabetically and after it in reality. Parse `DD.MM.YYYY` before sorting, everywhere.
- **Attendance is high** — most students are present on most sheets. A histogram of attendance percentage will be a spike near 100%. Say so, and choose charts that still communicate: the timeline and the heat map carry the story here, the distribution histogram does not. Choosing the right chart *and explaining why* is the marked skill.
- **`infovis.py` calls `charts.show_student(index, save_only)` and `charts.show_all(save_only)`** — §6.5, module-level functions. Your brief describes an `AttendanceCharts` class; keep it, and have the two functions wrap it.
- **`data/ground_truth.csv` already exists** with 30 labelled rows and a `note` column. Your accuracy report reads it directly.

### Contract

```python
# src/viz/style.py
def apply_house_style() -> None: ...
COLOURS = {"present": "#2E7D32", "absent": "#C62828", "uncertain": "#F9A825"}

# src/viz/charts.py
class AttendanceCharts:
    def __init__(self, db: Database): ...
    def student_timeline(self, index: str) -> Figure: ...
    def student_summary(self, index: str) -> Figure: ...
    def student_dashboard(self, index: str) -> Figure: ...
    def class_heatmap(self) -> Figure: ...
    def sheet_totals(self) -> Figure: ...
    def attendance_distribution(self) -> Figure: ...
    def save(self, fig: Figure, name: str) -> Path: ...

def show_student(index: str, save_only: bool = False) -> None: ...   # §6.5
def show_all(save_only: bool = False) -> None: ...                   # §6.5
```

`save_only=True` writes to `outputs/charts/` and opens no window — needed so the whole prototype can run headless and in tests.

`src/config.py` under `# --- M9 visualisation ---`:

```python
CHART_DPI = 150
CHART_STYLE = "seaborn-v0_8-whitegrid"
CHART_FIGSIZE = (11, 7)
```

### Tasks

**T1 — House style first.** Consistent fonts, a colour-blind-safe palette, no chartjunk, no 3-D. Every chart in the project uses it; that consistency is itself worth marks. Red/green alone fails colour-blind readers — add shape or position as a second cue and say so.
- `feat(viz): add shared chart style and colour palette`
- `docs(viz): note colour choices and accessibility reasoning`

**T2 — Student timeline.** Dates in chronological order on x, present/absent on y, step line with coloured markers, student name and index in the title.
- `feat(viz): add per student attendance timeline chart`
- `fix(viz): sort sheet dates chronologically not alphabetically`

**T3 — Student summary.** Attendance percentage as a donut with `4 / 5` printed in the centre, **plus the class average for context**. A number without a comparison is not a visualisation.
- `feat(viz): add attendance percentage donut chart`
- `feat(viz): add class average reference to the summary chart`

**T4 — The dashboard.** What `infovis.py 10000409` shows: `GridSpec`, four panels — timeline, donut, this student against the class, and a small table of date / present / confidence.
- `feat(viz): add multi panel student dashboard with gridspec`
- `feat(viz): add attendance details table panel`
- `fix(viz): tighten layout so panels do not overlap`

**T5 — Class-wide charts.** Heat map (6 students × 5 dates, cells labelled), present count per sheet, attendance distribution. **The heat map is the most useful chart a lecturer gets — make it the best one.**
- `feat(viz): add class attendance heatmap`
- `feat(viz): add present count per sheet bar chart`
- `feat(viz): add attendance percentage distribution histogram`

**T6 — Wire up `infovis.py` with M1.** Implement `show_student` and `show_all` to the §6.5 signatures. M1's CLI already handles argument parsing, the unknown-index message and the empty-database message — do not duplicate them.
- `feat(viz): add show_student and show_all entry points for the cli`
- `feat(viz): honour save-only by writing charts without opening a window`

**T7 — End-to-end runner (QA).** `tools/run_all_sheets.py`: wipe the DB, run all five sheets, catch and report failures, print rows detected / students matched / time per sheet. **Run it after every merge into `main` and tell the group the same hour when it breaks.**
- `feat(qa): add end to end runner for all five sheets`
- `feat(qa): report per sheet timing and detected row counts`
- `fix(qa): reset database before each full run`

**T8 — Accuracy against ground truth, with M7.** Overall, per sheet, and **a list of every mistake with its student index and sheet date** so the group can open those exact cells. The two ink-but-absent cells will be in that list — name them.
- `feat(qa): compute accuracy against ground truth per sheet`
- `feat(qa): list individual misclassified cells for review`

**T9 — Report asset generator (last).** `tools/make_report_assets.py` regenerates **every** figure in `outputs/` in one command, so a late parameter change does not mean re-taking screenshots by hand.
- `feat(report): add single command to regenerate all report figures`
- `docs(report): list every generated figure and its report section`

**Verify:** `python infovis.py 10000409` shows and saves the dashboard; `python infovis.py --all` draws the class charts; `python tools/run_all_sheets.py` passes on all five sheets.

**Figures:** `m9_dashboard_example.png`, `m9_class_heatmap.png`, `m9_attendance_distribution.png`, `m9_accuracy_report.png`, `m9_chart_type_choice.png` (**the same data as a pie, a bar and a timeline, showing why you chose what you chose — worth real marks in a visualisation module**)

---

## 10. Error handling rules

- User mistakes (missing file, bad index, empty database) → friendly one-line message, `sys.exit(2)`, **no traceback**.
- Programming errors → full traceback, wrapped so the stage name is visible.
- Never use a bare `except:`. Never `except Exception: pass`.
- Warnings that do not stop the run (for example, row count ≠ student count) go through the logger at `WARNING`, and appear in the final summary.

---

## 11. Tests each module must write

*"Testing results"* is named in the marking criteria, so this is not optional. Every test file starts with `matplotlib.use("Agg")` before any pyplot import — the suite must never try to open a window. Prefer small synthetic images over the real sheets: a test that needs a 3024 x 4032 photo is slow and tells you less.

**M1 — `tests/test_pipeline.py`**
1. Two fake stages append to a list — `Pipeline` runs them in the given order.
2. A stage that raises → the error message contains that stage's `name`.
3. `ProgressViewer.save_all()` into `tmp_path` writes exactly N files, numbered from 01.
4. `ProgressViewer.add()` accepts both a 3-channel and a 1-channel image.
5. `sams.py` with a missing image exits with code 2 (use `subprocess`).
6. `Student("007", "x").index == "007"` — leading zero survives.

**M2 — `tests/test_geometry.py`**
1. `load_image` raises `FileNotFoundError` on a bad path, `ValueError` on a non-image.
2. A synthetic white rectangle on a dark background, rotated 7°, is corrected to within 1°.
3. `four_point_warp` on a known quad gives the expected output size.
4. Corner ordering returns top-left first for shuffled input points.
5. The fallback path triggers on pure noise and does not crash.

**M3 — `tests/test_enhance.py`**
1. `to_grey` output is 2-D `uint8`, same height and width as the input.
2. Luminosity of a pure red pixel `(0, 0, 255)` BGR is ≈ 76.
3. `denoise` reduces variance on a synthetic noisy flat patch.
4. `remove_shadow` on a linear brightness ramp reduces the left-half / right-half difference.
5. An unknown method name raises `ValueError`.

**M4 — `tests/test_binarize.py`**
1. Output is strictly two-valued: `set(np.unique(out)) <= {0, 255}`.
2. **Ink is 255** — a dark stroke on white paper comes out white.
3. Hand-written Otsu on a clean two-peak image lands between the peaks and matches OpenCV within ±1.
4. Opening removes isolated single pixels; closing fills a 1-pixel gap in a line.
5. An even `block` value is corrected or raises a clear `ValueError`.

**M5 — `tests/test_table.py`**
1. A synthetic drawn grid: the detector finds exactly the lines that were drawn.
2. With one horizontal line erased, grid repair restores it at the right position.
3. `Grid.cell_bbox(0, 4)` returns the expected box for a known grid.
4. Lines 3 px apart merge into one; lines 40 px apart do not.
5. **Two stacked synthetic tables — the selector returns the taller lower one**, not the header.
6. `crop_cell` output is smaller than the raw bbox by `2 * CELL_INSET` horizontally.

**M6 — `tests/test_ink.py`**
1. A synthetic white cell with a blue stroke: the mask covers it, `ink_ratio > 0`.
2. A blank white cell: `ink_ratio` near zero, `components == 0`.
3. A **black** stroke is detected — the low-saturation branch works.
4. A **red** stroke is detected — hue wrap-around at 0/180 is handled.
5. A **green** stroke is detected (no green pen exists in the real data, so this test is the only proof).
6. A 4-pixel speck is removed by `MIN_BLOB_AREA`; a 200-pixel stroke is kept.
7. `dominant_pen_colour` returns `"blue"` for a pure blue stroke.

**M7 — `tests/test_decision.py` and `tests/test_db.py`**
1. `decide` returns `True` for a high ink ratio, `False` for near zero.
2. A value just below the threshold returns `False` with low confidence.
3. Many tiny components with low stroke length is rejected — a smudge is not a signature.
4. `init_schema` on a temp file creates all four tables.
5. Saving the same sheet twice does not duplicate attendance rows.
6. `get_attendance(index)` returns rows in **date** order, not string order.
7. A SQL injection attempt in a student index changes nothing — proves parameterised queries.
8. `parse_students` keeps `"10000409"` as a string, and finds all six students via `.//student`.

**M8 — `tests/test_recognition.py`**
1. `compare(x, x)` gives a combined score near 1.0.
2. A signature against a blank image scores low.
3. The same signature shifted 10 px still scores high — centring works.
4. The same signature scaled 1.5× still scores high — scale normalisation works.
5. `normalise_signature` always returns exactly `SIG_NORM_SIZE`.
6. A student with 1 sample returns a clear "not enough samples" result, no exception.

**M9 — `tests/test_charts.py` and `tests/test_e2e.py`**
1. Each chart function returns a `matplotlib.figure.Figure`.
2. A student with no records produces an empty-state chart, not a crash.
3. Dates are ordered chronologically — assert `05.07.2019` comes **after** `21.06.2019`.
4. `save()` writes a file of non-zero size.
5. The full pipeline on one sheet writes the expected number of attendance rows.
6. Running the same sheet twice leaves the row count unchanged.

Run the whole suite from the repository root with `pytest -q`. It must pass on `main` at all times; if your merge breaks someone else's test, that is yours to fix.

---

## 12. Commit protocol

- Branch: `feat/m<N>-<area>` — `feat/m1-core`, `feat/m2-geometry`, `feat/m3-enhance`, `feat/m4-binarize`, `feat/m5-table`, `feat/m6-ink`, `feat/m7-decision-db`, `feat/m8-recognition`, `feat/m9-viz-qa`.
- Merge into `main` via PR with a **merge commit or rebase — never squash.** Squashing collapses your fifteen commits into one and erases the evidence of your individual contribution.
- Message format: `type(scope): short lowercase summary`, types `feat | fix | refactor | test | docs | chore`.
- One logical change per commit. **Minimum 15 commits each, 20 for M1.** Push after each task, not in one batch at the end — a single dump on the last night is visible in the log and is worth close to nothing.
- Author identity must be your own, set **before your first commit**:
  ```fish
  git config user.name "Your Full Name"
  git config user.email "<your index>@students.nsbm.ac.lk"
  ```
- Never force-push. Never rewrite shared history. `git log --author` is the evidence of individual contribution, which is 15% of the marks.
- Nobody commits directly to `main` except M1, and only for integration.

---

## 13. Definition of done

**Every module, without exception:**

- [ ] Your stage runs inside `sams.py` on **all five sheets** without crashing
- [ ] Your tunables are in your own block in `src/config.py`, nothing hard-coded
- [ ] `pytest -q` passes from the repository root
- [ ] Your figures are in `outputs/figures/m<N>_*.png` at 150 dpi or better
- [ ] 15+ commits on your branch with clear messages
- [ ] `docs/contrib_m<N>.md` drafted — two pages, per the coursework brief

**Plus, per module:**

| | Also done when |
|---|---|
| **M1** | §4 facts measured and recorded · all three commands run with `--help` and fail gracefully · summary table and numbered step images produced · montage displays every step · `README.md` complete and dependencies pinned · **all stubs deleted, `grep -r "STUB" src/` returns nothing** · 20+ commits |
| **M2** | all five sheets produce a straight, tightly cropped `warped` image with **both tables and all five columns intact** · nothing crashes when corners are not found · the path taken is logged per sheet |
| **M3** | `grey` is evenly lit on all five sheets · shadow removal visibly helps at least one problem sheet · settings agreed with M4 **and** M5 and recorded in `config.py` |
| **M4** | `binary` is two-valued with ink 255 on all five sheets · hand-written Otsu matches OpenCV within ±1 · **M5 confirms the table lines survive your morphology** |
| **M5** | correct counts on all five sheets: **6 data rows, 5 columns** · the lecture header table is identified and discarded · `cells` gives M6 clean colour crops of the signature column · grid repair fixes at least one real missing line |
| **M6** | blue, black, red and green ink all detected (green and black proven synthetically) · the three genuinely empty cells give near-zero ink ratio · crops and masks saved as `<index>.png` for M8 · feature key names agreed with M7 |
| **M7** | all five sheets in the database with no duplicates on re-run · threshold set from a **sweep against `ground_truth.csv`**, not guessed · accuracy reported both including and excluding the two ink-but-absent cells · M8 and M9 can read from the database |
| **M8** | `investigate.py <index>` works for every student · `MATCH_THRESHOLD` set from the measured EER, not guessed · shift and scale invariance tests pass · the 41-pair sample size stated honestly |
| **M9** | `infovis.py <index>` shows and saves the dashboard · `--all` draws the class charts · `run_all_sheets.py` passes on all five · accuracy measured and **every** misclassified cell listed by index and date · `make_report_assets.py` regenerates everything in one command |

**Group, before submission:**

- [ ] `python tools/run_all_sheets.py` green on all five sheets
- [ ] `grep -r "STUB" src/` returns nothing
- [ ] `pytest -q` passes on a clean clone in a fresh venv
- [ ] All nine `docs/contrib_m*.md` written
- [ ] Report front page lists every member's name, role and index
- [ ] `v1.0` tagged

---

## 14. Integration order and hand-offs

Who unblocks whom. Nobody waits for a module that is not directly above them — stubs and fixtures cover the gap (§8).

```
M1 skeleton + contracts + fixtures     ← everybody waits on this, and only this
   │
   ├── M2 geometry → M3 enhance → M4 binarise → M5 table → M6 ink → M7 decision
   │                                                                    │
   │                                                          ┌─────────┴─────────┐
   │                                                          ▼                   ▼
   └── M7 database schema + seed_db.py ──────────────────▶ M9 charts        M8 recognition
```

| Hand-off | From | To | What has to be true |
|---|---|---|---|
| skeleton, `Stage`, `models`, `config` | M1 | everyone | on `main` before anyone starts |
| `data/fixtures/*.png` | M1 | M3–M8 | crude but type-correct, so nobody idles |
| `ctx["warped"]` | M2 | M3, M5 | flat, tightly cropped, both tables visible |
| `ctx["grey"]` | M3 | M4 | evenly lit; settings agreed jointly with M4 |
| `ctx["binary"]` | M4 | M5 | ink 255, table lines intact after morphology |
| `ctx["cells"]` | M5 | M6 | signature column only, **colour** crops, `.row` set, no gaps |
| `ctx["ink"]` | M6 | M7 | one `InkResult` per cell, same order; key names agreed with M7 first |
| `outputs/cells/<date>/<index>.png` | M6 | M8 | crop and `_mask` saved per present student |
| database schema + `tools/seed_db.py` | M7 | M8, M9 | **pushed first, before M7's own decision logic** |
| `ctx["records"]` | M7 | M1 | drives the summary table |
| `charts.show_student` / `show_all` | M9 | M1 | §6.5 signatures exactly |
| `matcher.investigate` | M8 | M1 | §6.5 signature exactly |

**Integration is M1's job, one module at a time.** As each lands: delete its stub from `src/stubs.py`, swap the line in `STAGES`, run `sams.py` on all five sheets, commit `refactor(pipeline): replace <stage> stub with real implementation`. Never swap two at once — when it breaks you will not know which one did it.

### The three decisions the group has to make together

Each has an owner and a deadline of *before that module is merged*.

1. **Overflowing signatures — M5 and M6 agree the approach.** §4 deviation 6. A signature that spills into the row below belongs to the row it *starts* in. Proposal: M5 crops with a small vertical pad, M6 attributes each connected component to the row holding most of its pixels. Whoever writes it first tells the other.
2. **Ink that is not a signature — M6 supplies, M7 decides. Measured, see below.** §4 deviation 7. `ab` and a stray tick are both ink and both mean absent. M6's features now all reach M7 on `InkResult`, and the separation has been measured across all 30 cells:

   | | n | ink_ratio | filled_ratio | centroid_offset | aspect |
   |---|---|---|---|---|---|
   | genuine signatures | 25 | 0.207 | 0.362 | 48.6 | 4.47 |
   | empty cells | 3 | 0.006 | 0.106 | 25.6 | 1.77 |
   | **ink but absent** | 2 | **0.507** | **0.752** | 44.4 | 4.30 |

   The two awkward cells are `21.06.2019 / 10009306` (the lecturer's `ab`, `filled_ratio` 0.740) and `05.07.2019 / 10009303` (a stray red tick, 0.765).

   **No single feature separates them.** The best is `filled_ratio`, and rejecting above the lower awkward value still loses 2 of the 25 genuine signatures. `ink_ratio` loses 7 — the `ab` has a *higher* ink ratio than any real signature on any sheet.

   One pair does separate cleanly: `filled_ratio >= 0.74 AND stroke_length >= 192` rejects both and keeps all 25. **Treat that as a measurement, not an answer.** It is fitted to two positive examples and its thresholds sit exactly on their values, so it is memorisation rather than a rule. M7 decides whether to use it with that caveat stated, or to report the two cells as a known limitation. Either is defensible in the report; presenting the pair as a validated rule is not.
3. **The confidence band — M7 with M1.** `config.UNCERTAIN_BELOW` decides what the summary calls uncertain. M7 sets it from the threshold sweep, not by feel.

---
