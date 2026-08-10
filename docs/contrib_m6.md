# M6: Ink Segmentation Contribution Notes

Module: `src/detect/`
Branch: `feat/m6-ink`
Files: `cell_clean.py`, `ink_mask.py`, `tests/test_ink.py`

## What this module does

Processes cell crops produced by M5: cleans leftover printed border lines, removes outer margin artifacts, segments pen ink (blue, black, red, green) from white paper background across HSV and LAB colour spaces, cleans masks via connected component analysis, extracts feature vectors for M7 (present/absent decision), and saves crops and masks for M8 signature recognition.

## Technique & Colour Space Analysis: HSV vs LAB Masking

Handling multi-colour pens (blue, black, red, green) is the headline contribution of Module M6. Two major colour spaces were implemented and evaluated: **HSV (Hue-Saturation-Value)** and **CIELAB ($L^*a^*b^*$)**.

### 1. HSV Segmentation (Winning Method)
- **Why RGB is Poor, HSV is Superior**: In RGB, lightness and chromaticity are tightly coupled; shadows or bright spots shift all three R, G, and B channels simultaneously. HSV isolates colour (Hue and Saturation) from illumination (Value).
- **Coloured Pens (Blue, Green, Red)**: White paper background has near-zero saturation ($S \approx 0$). Coloured pen inks have high saturation ($S \ge \text{SAT\_MIN} = 50$). This produces clean, shadow-resistant segmentation of coloured strokes.
- **Red Pen Handling**: Red hue wraps around in OpenCV HSV space ($[0, 10]$ and $[170, 180]$). Saturation thresholding avoids complex hue split logic for detection, while hue ranges map wrapped red hues accurately during pen colour classification.
- **Black Pen Handling**: Black pen ink breaks the saturation rule because black ink has low saturation similar to paper. Black ink is detected via low value/lightness ($V \le \text{VAL\_MAX} = 200$).
- **Combined HSV Mask**:
  $$\text{Mask}_{\text{HSV}} = (S \ge \text{SAT\_MIN}) \lor (V \le \text{VAL\_MAX})$$

### 2. CIELAB ($L^*a^*b^*$) Segmentation
- **Coloured Pens**: Paper background is neutral achromatic ($a^* \approx 128, b^* \approx 128$). Chrominance distance from neutral gray is calculated as:
  $$\text{chroma} = \sqrt{(a^* - 128)^2 + (b^* - 128)^2}$$
  Coloured pens are detected where $\text{chroma} \ge 12.0$.
- **Black Pen**: Detected where luminance $L^* \le \text{VAL\_MAX}$.

### Comparison & Evaluation Results
- **Illumination Sensitivity**: While LAB $a^*, b^*$ channels decouple lightness from colour, faint lighting gradients and uneven paper yellowing on actual phone captures induce minor shifts in neutral $a^*, b^*$ baselines.
- **Paper Noise**: Low $L^*$ thresholding in LAB space picked up subtle paper texture shadows under uneven lighting.
- **Verdict**: **HSV combined segmentation won**. $S \ge 50$ provides an extremely sharp cut-off between white paper and coloured ink, and $V \le 200$ captures black pen ink without picking up paper background.

## Failure Modes & Edge Cases Analysis

Across the 5 test sheets, three specific failure cases and difficult edge cases were encountered and addressed:

1. **Faded Ink and Light Pencil**:
   - *Failure Case*: Light ballpoint ink or faint HB pencil has moderate saturation ($S \approx 55$) and lighter value ($V \approx 205$), risking false negatives.
   - *Mitigation*: Lowered `SAT_MIN` from 60 to 50 and set `VAL_MAX = 200` to catch faint strokes without crossing paper background texture thresholds.

2. **Signatures Crossing Row Boundaries**:
   - *Failure Case*: On sheets `31.05.2019` and `05.07.2019`, tall signature loops cross vertical row rules into adjacent rows.
   - *Mitigation*: M5 applies `CELL_PAD_Y = 6` to capture overflowing strokes. M6 computes `stroke_bbox` and `centroid_offset` relative to the expanded crop to maintain full feature representation.

3. **Printed Dots vs. Ink Specks**:
   - *Failure Case*: Stray paper specks or tiny printed grid artifacts can be misidentified as student signature ink.
   - *Mitigation*: Connected component analysis filters components smaller than `MIN_BLOB_AREA = 12` pixels. Sparse signatures have low `filled_ratio`, whereas compact printed dots or smudges have high `filled_ratio`, allowing M7 to differentiate them.

## Problems and how I solved them

1. **Leftover Table Borders inside Cell Crops**:
   - *Problem*: Even with cell insets, printed table rules surviving along cell boundaries were detected as dark ink.
   - *Solution*: `remove_table_lines` uses horizontal and vertical morphological opening kernels near cell margins and clears the outer 2-pixel frame.

2. **Broken Pen Strokes**:
   - *Problem*: Fast pen flourishes leave micro-gaps within a single continuous stroke.
   - *Solution*: `close_stroke_gaps` applies morphological closing with a $3 \times 3$ rectangular kernel to join broken strokes without inflating overall ink ratio.
