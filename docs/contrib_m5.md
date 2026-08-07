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
Every sheet has a one-row lecture header table above the student table.
`select_student_table` groups horizontal lines into bands separated by gaps
larger than 40 px. The band with the most lines is the student table. The
other band is logged and discarded. If only one band is found a WARNING is
issued.

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

On all five sheets: `len(ctx["cells"]) == 6` and every crop in
`m5_cells_numbered.png` shows a signature box, not a name column.
