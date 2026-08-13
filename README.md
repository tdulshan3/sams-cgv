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

| | Module | Owner | Index |
|---|---|---|---|
| M1 | Lead, contracts, pipeline, three CLIs, step viewer | T.R.D.T. Dulshan | 28994 |
| M2 | Acquisition & geometry | J M S V Jayaweera | 29008 |
| M3 | Greyscale & enhancement | K.A.A.T Kulasooriya | 28282 |
| M4 | Binarisation & morphology | H.P.G Dilina Mewan | 28775 |
| M5 | Table detection | B G D M Beligala | 28232 |
| M6 | Ink segmentation | K D R Laksika | 28371 |
| M7 | Decision & database | Uduwana T M | 29318 |
| M8 | Signature recognition | KGP Kalhara | 28176 |
| M9 | Visualisation & QA | P R A D Pethiyagoda | 29058 |

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
