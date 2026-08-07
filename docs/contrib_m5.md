# M5 — Table Detection Contribution Notes

Module: `src/table/`
Branch: `feat/m5-table`
Files: `line_detect.py`, `grid_builder.py`, `cell_extract.py`, `tests/test_table.py`

## What this module does

Reads `ctx["binary"]` and `ctx["warped"]`, finds the student signing table,
and writes `ctx["grid"]` (line positions) and `ctx["cells"]` (6 signature
cell crops from the colour warped image).

## Key decisions

### Two tables on the page
Every sheet has a one-row lecture header table above the student table, and it
carries a signature of its own — the lecturer's. Selecting the wrong table
reports that signature as a student's, and the output still looks plausible.

My first approach grouped horizontal lines into bands separated by gaps larger
than 40 px and took the band with the most lines. **Measured against the real
sheets, that cannot work.** The student rows sit ~44 px apart and the gap
between the two tables is only ~64 px, so every threshold either shatters the
student table into single-line bands or swallows the header table with it. On
three of five sheets it produced one row instead of six.

`longest_regular_run` replaces it. A ruled table is regular by construction —
its rows are the same height — while the lines around it are not, so the
student table is the longest stretch of horizontal lines whose gaps all sit
within `ROW_SPACING_TOLERANCE` of that stretch's own median gap. That separates
the two tables by a property they genuinely differ in rather than by a pixel
distance that happens to fall between them.

One wrinkle: a printed rule thick enough to produce two projection peaks
arrives as two positions ~9 px apart, and those near-duplicates break the run
where they appear — on `12.07.2019` that truncated the table at four rows.
`_merge_closer_than` collapses anything closer than `MIN_ROW_HEIGHT` first,
since no row is that short.

### Columns are detected inside the table, not across the page
Searching the whole page for vertical rules also finds the header table's
columns, the sheet edge and the edge of the photograph. They arrive as extra
entries at both ends of `xs`, and because the signature column is addressed
**by index**, two spurious lines on the left silently shift every column: six
cells are still produced and every one holds a student's printed *name*.

Two changes fix it. Vertical detection runs on the student band only, which
also makes `V_KERNEL_RATIO` mean what it was supposed to — a fraction of the
table's own height, not of the whole page, which is why a 0.30 kernel had been
eroding every column rule away. Then `_clip_to_table_width` drops any vertical
outside the span of the row rules, since the row rules span exactly the table's
width and nothing else.

### Primary method: morphology
`line_mask` erodes then dilates with a long axis-aligned kernel. Only ink
runs longer than `H_KERNEL_RATIO`/`V_KERNEL_RATIO` of the image dimension
survive. The result is summed perpendicular to the expected line direction
(projection profile) and `scipy.signal.find_peaks` locates the peaks.
Peaks closer than `LINE_MERGE_TOL` px are merged by averaging.

Morphology was chosen over Hough as the primary method because Hough fires on
signature strokes and table-border intersections, producing a noisy peak
distribution on documents with heavy ink. The morphology approach suppresses
anything too short to be a table rule.

### Hough cross-check (T4)
`detect_lines_hough` runs `HoughLinesP` and keeps lines within 5 degrees of
axis-aligned. The counts are logged alongside the morphology counts at every
run so any divergence is visible without additional tooling. The Hough
positions are not used to build the grid.

### Grid repair (T5)
`_repair_missing_lines` computes the median inter-line spacing and checks each
gap. A gap close to a multiple of the median indicates a missing line.
Intermediate positions are inserted and a WARNING naming the gap is logged.
`_drop_too_close` removes duplicates before repair.

### Cell cropping
`crop_cell` shaves `CELL_INSET` (4 px) off each side to exclude the printed
border, and extends `CELL_PAD_Y` (6 px) below the nominal bottom to catch
signatures that overflow. Crops come from `ctx["warped"]` (colour), never from
`ctx["binary"]`, because M6 uses pen colour in its HSV/LAB ink masks.

### Column count correction
The original brief assumed 4 columns. The real sheet has 5:
`No | Student No | Title | Student Name | Signature`
`SIGNATURE_COL = 4` is always used; the literal `4` never appears in M5 code.

## Parameters in config.py (M5 block)

| Name | Value | Why |
|---|---|---|
| `H_KERNEL_RATIO` | 0.30 | 30% of width: table rules are ≥ 50% wide; 30% cuts short strokes |
| `V_KERNEL_RATIO` | 0.30 | Same logic for column borders |
| `LINE_MERGE_TOL` | 8 | A printed rule can appear as 2–5 px wide in the binary |
| `MIN_ROW_HEIGHT` | 18 | Shortest plausible row on a resized 1600-px sheet |
| `MIN_COL_WIDTH` | 25 | Shortest plausible column |
| `CELL_INSET` | 4 | Excludes the border without biting into the signature |
| `CELL_PAD_Y` | 6 | Keeps overflow from `31.05.2019` and `05.07.2019` |
| `EXPECTED_COLS` | 5 | Measured T0 |
| `EXPECTED_DATA_ROWS` | 6 | Measured T0 |

## Figures produced

| File | Contents |
|---|---|
| `m5_line_masks.png` | H and V morphology masks side by side |
| `m5_projection_profiles.png` | Column/row sum graphs with detected peaks |
| `m5_grid_overlay.png` | Green verticals + red horizontals over the warped sheet |
| `m5_cells_numbered.png` | 6 signature crops labelled 0–5 |
| `m5_grid_repair.png` | Before/after showing an inserted line |
| `m5_two_tables.png` | Both table bands coloured differently |
| `m5_hough_vs_morphology.png` | Side-by-side comparison |

## Verification

On all five sheets the grid comes out as 6 data rows × 5 columns and produces
6 signature cells, each a colour crop of column 4:

| Sheet | rows | cols | cells | crop size |
|---|---|---|---|---|
| 31.05.2019 | 6 | 5 | 6 | 232 × 43 |
| 21.06.2019 | 6 | 5 | 6 | 224 × 41 |
| 28.06.2019 | 6 | 5 | 6 | 221 × 42 |
| 05.07.2019 | 6 | 5 | 6 | 227 × 42 |
| 12.07.2019 | 6 | 5 | 6 | 227 × 45 |

`tests/test_real_sheets.py` asserts this on every run, including a cross-check
that the signature cell starts further right than every other column — so an
off-by-one in the grid fails the build even if the column *count* is right.

## What I learned about testing

My module shipped with 84 passing tests and produced **zero cells on every real
sheet**. All 84 were synthetic. Two things hid behind them:

* `cv2.HoughLinesP` returns `(N, 4)` on the OpenCV 5 we pin and `(N, 1, 4)` on
  OpenCV 4, so `lines[:, 0]` crashed on any image with enough lines to return a
  result — which a small synthetic fixture never has. It is now wrapped once in
  `src/utils/cvcompat.py`.
* Upstream, M2's geometry was handing me mirrored 412 × 52 crops — 0.6% of the
  page. I had been tuning line detection against images with no table in them.

The lesson is that a synthetic fixture tests the code I wrote against the input
I imagined. Neither defect was visible until `sams.py` was run on a real photo,
and that run is now part of the suite.

---

*Integration note: the band-selection rewrite, the column clipping, the
duplicate-rule merge and the `Grid` bounds guards were applied by M1 during
integration under deadline pressure, and are described above as they now stand
in the code. Review them before submission and rewrite this section in your own
words.*
