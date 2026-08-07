# SAMS — Student Attendance Management System

CS402.3 Computer Graphics and Visualization — NSBM Green University.

A university records attendance on paper signing sheets. An admin photographs a
sheet with a phone and hands over the picture plus a student list. SAMS reads
the photo, works out which signature boxes were signed and which were left
empty, stores that as attendance in a local SQLite database, and draws graphs
of it. It also compares one student's signatures across sheets and flags one
that does not look like the others.

Command line only. There is no web page and no desktop app — the brief fixes
three commands, and those are the whole interface.

---

## What state this is in

The pipeline, the three programs, the step-by-step viewer, and the shared
contracts (M1) are finished. The Acquisition & Geometry Correction stage (M2) 
is also completed and integrated. The remaining six image processing modules 
are owned by other members of the group and are still standing in as placeholders, 
so every command runs end to end today but reports zero students. Each placeholder
announces itself:

```
[22:37:22] WARNING stubs     | STUB binarize: returning placeholder data (owned by M4)
```

As each module lands it replaces one line of the `STAGES` list in `sams.py` and
one class in `src/stubs.py`. Nothing else changes.

---

## Install

Python 3.11 or newer. On Arch based systems do not install into the system
Python — it is externally managed.

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
`12.07.2019`, and the real student indices are eight digits — see section 4 of
`BUILD_SPEC.md`.

### `sams.py` — process one sheet

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
 Duration  : 0.19 s
 Steps     : outputs/steps/12.07.2019/  (4 images)
─────────────────────────────────────────────────────────────
```

`--no-show` skips the montage window, `--no-save` skips writing step images,
`--debug` turns on verbose logging.

### `infovis.py` — attendance charts

```
usage: infovis.py [-h] [--all] [--save-only] [--debug] [index]
```

One student, or `--all` for the class. `--save-only` writes to
`outputs/charts/` without opening a window.

### `investigate.py` — compare signatures

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
  stubs.py        placeholder stages, deleted as each real module lands
  utils/          Stage base class, logging, timing
  io/             image loading, info.xml parsing, the database
  preprocess/     geometry, enhancement, binarisation
  table/          line detection, grid building, cell extraction
  detect/         ink segmentation and the present or absent decision
  recognise/      signature comparison
  viz/            the step viewer and the charts
tools/            input inspection, bootstrap fixtures, report figures
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

* Six of the image processing modules are placeholders, so attendance counts are
  zero until they land.
* `info.xml` was not supplied with the sheet photos. It is reconstructed from
  Figure 1 of the brief and the printed student table. The brief's own example
  is not well-formed XML — a tag may not start with a digit — so the batch is
  carried as an attribute instead.
* Signatures routinely run past their cell borders, and on two sheets a cell
  holds ink that is not a signature: a handwritten `ab` on `21.06.2019` and a
  stray red mark on `05.07.2019`. Both mean absent. Ink ratio alone gets them
  wrong, and handling them is an open question in `BUILD_SPEC.md` section 14.
* Step images are downscaled to 1400 px wide before being written. The full
  size photos are 3024 x 4032 and nothing in the report is printed that large.
