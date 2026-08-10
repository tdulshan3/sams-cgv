# SAMS: Student Attendance Management System

CS402.3 Computer Graphics and Visualization, NSBM Green University.

A university records attendance on paper signing sheets. An admin photographs a
sheet with a phone and hands over the picture plus a student list. SAMS reads
the photo, works out which signature boxes were signed and which were left
empty, stores that as attendance in a local SQLite database, and draws graphs
of it. It also compares one student's signatures across sheets and flags one
that does not look like the others.

Command line only. There is no web page and no desktop app; the brief fixes
three commands, and those are the whole interface.

---

## What state this is in

**Complete.** All nine modules are merged and wired in, every placeholder has
been deleted, and the three commands run end to end on all five sheets.

| | Module | Owner |
|---|---|---|
| M1 | Lead, contracts, pipeline, three CLIs, step viewer | Dulshan |
| M2 | Acquisition & geometry | Jayaweera |
| M3 | Greyscale & enhancement | Kulasooriya |
| M4 | Binarisation & morphology | Dilina |
| M5 | Table detection | dushaDev |
| M6 | Ink segmentation | Laksika |
| M7 | Decision & database | Malinda |
| M8 | Signature recognition | Kalhara |
| M9 | Visualisation & QA | Pethiyagoda |

Measured against `data/ground_truth.csv`, which was transcribed by eye from the
five sheets:

**Attendance accuracy: 29 of 30 cells, 96.7%.**

| Sheet | Accuracy |
|---|---|
| 31.05.2019 | 6/6 |
| 21.06.2019 | 6/6 |
| 28.06.2019 | 6/6 |
| 12.07.2019 | 6/6 |
| 05.07.2019 | 5/6 |

The single miss is `05.07.2019 / 10009303`, where a stray red tick sits under a
signature overflowing from the row above. The system reports it at confidence
0.55, below the uncertainty threshold, so it appears in the summary's
`Uncertain` count rather than being asserted as fact.

---

## Install

Python 3.11 or newer. On Arch based systems do not install into the system
Python: it is externally managed.

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

Bash or zsh: `source .venv/bin/activate`.

---

## The three commands

```bash
python sams.py data/sheets/12.07.2019.png data/info.xml
python infovis.py 10000409
python investigate.py 10000409
```

The brief illustrates these with `10.07.2019.png` and index `001`. Our five
sheets are dated `31.05.2019`, `21.06.2019`, `28.06.2019`, `05.07.2019` and
`12.07.2019`, and the real student indices are eight digits, see section 4 of
`BUILD_SPEC.md`.

### `sams.py`, process one sheet

```
usage: sams.py [-h] [--no-show] [--no-save] [--debug] image xml
```

Runs the six stages in order, shows each step as it happens, writes the
numbered step images, and prints a summary:

```
─────────────────────────────────────────────────────────────
 Sheet     : 12.07.2019
 Students  : 6
 Present   : 6
 Absent    : 0
 Uncertain : 0
 Duration  : 0.57 s
 Steps     : outputs/steps/12.07.2019/  (17 images)
─────────────────────────────────────────────────────────────
```

`--no-show` skips the montage window, `--no-save` skips writing step images,
`--debug` turns on verbose logging.

### `infovis.py`, attendance charts

```
usage: infovis.py [-h] [--all] [--save-only] [--debug] [index]
```

One student, or `--all` for the class. `--save-only` writes to
`outputs/charts/` without opening a window.

### `investigate.py`, compare signatures

```
usage: investigate.py [-h] [--save-only] [--debug] index
```

Needs at least two saved signature crops for that student, so process a few
sheets first.

---

## Folder map

```
sams.py  infovis.py  investigate.py   the three programs
BUILD_SPEC.md                         the contract every module builds against
src/
  config.py       every tunable number in the project
  models.py       Student, SheetMeta, Cell, InkResult, AttendanceRecord
  pipeline.py     runs the stages in order, times them, names the one that failed
  cli.py          shared helpers for the three programs
  utils/          Stage base class, logging, timing, opencv 5 wrappers
  io/             image loading, info.xml parsing, the database
  preprocess/     geometry, enhancement, binarisation
  table/          line detection, grid building, cell extraction
  detect/         ink segmentation and the present or absent decision
  recognise/      signature comparison
  viz/            the step viewer and the charts
tools/            input inspection, report figures, QA runner, threshold sweeps
tests/            pytest suite
data/
  sheets/         the five signing sheet photos, named by the date on the sheet
  info.xml        student and subject records
  ground_truth.csv  what a human sees on each sheet, for measuring accuracy
  attendance.db   written by sams.py, not committed
outputs/          every image and figure the run produces, not committed
docs/             per member contribution notes
```

## How it fits together

Every processing step is a subclass of `Stage` with one method, `run(ctx)`. The
`Pipeline` owns the order and nothing else: it times each stage, collects the
pictures the stage wants shown, and if a stage raises it stops the run and says
which stage it was. That is what lets nine people replace one step at a time
without touching anything around it.

```
photo ─▶ geometry ─▶ enhance ─▶ binarize ─▶ table ─▶ ink ─▶ decision ─▶ attendance.db
```

The diagram version is at `outputs/figures/m1_architecture.png`, drawn by:

```bash
python tools/make_m1_figures.py 12.07.2019.png
```

## Tests

```bash
pytest -q
```

## Known limits

Stated plainly, because a measured limitation is worth more than a claim that
does not survive checking.

* **Signature matching barely works.** `investigate.py` runs, but the
  genuine-against-impostor experiment (`python tools/eval_recognition.py`)
  measures an **Equal Error Rate of 43%** across 60 genuine and 375 impostor
  pairs: close to the 50% of a coin toss. `MATCH_THRESHOLD` is set from that
  measurement rather than guessed, so the tool is at least internally honest,
  but it cannot reliably tell one student's signature from another's. The most
  likely causes are the sample size (5 per student, no known forgeries) and
  contamination: crops are ~225 x 43 px and neighbouring signatures bleed
  across the row borders into them.
* **Geometry never uses the perspective warp.** All five sheets are
  photographed on a pale desk, so the paper edge has too little contrast for
  contour detection and every sheet takes the rotation fallback instead. The
  result is straight, but it is not a true top-down correction.
* **One attendance cell in thirty is wrong**, see above. It is flagged
  uncertain rather than asserted.
* **`info.xml` was not supplied** with the sheet photos. It is reconstructed
  from Figure 1 of the brief and the printed student table. The brief's own
  example is not well-formed XML, a tag may not start with a digit, so the
  batch is carried as an attribute instead.
* **Ink is not always a signature.** A handwritten `ab` on `21.06.2019` and a
  stray red mark on `05.07.2019` both mean absent. The decision stage rejects
  the first on stroke shape and gets the second wrong.
* Step images are downscaled to 1400 px wide before being written. The full
  size photos are 3024 x 4032 and nothing in the report is printed that large.
