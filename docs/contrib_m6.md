# M6 — Ink Segmentation Contribution Notes

Module: `src/detect/`
Branch: `feat/m6-ink`
Files: `cell_clean.py`, `ink_mask.py`, `tests/test_ink.py`

## What this module does

Processes cell crops produced by M5: cleans leftover printed border lines, removes outer margin artifacts, segments pen ink (blue, black, red, green) from white paper background across HSV and LAB colour spaces, cleans masks via connected component analysis, extracts feature vectors for M7 (present/absent decision), and saves crops and masks for M8 signature recognition.

## Technique & Colour Space Analysis: HSV vs LAB Masking

Handling multi-colour pens (blue, black, red, green) is the headline contribution of Module M6. Two major colour spaces were implemented and evaluated: **HSV (Hue-Saturation-Value)** and **CIELAB ($L^*a^*b^*$)**.

### 1. HSV Segmentation (Winning Method)
- **Coloured Pens (Blue, Green, Red)**: White paper background has near-zero saturation ($S \approx 0$). Coloured pen inks have high saturation ($S \ge \text{SAT\_MIN} = 60$). This produces clean, shadow-resistant segmentation of coloured strokes.
- **Red Pen Handling**: Red hue wraps around in OpenCV HSV space ($[0, 10]$ and $[170, 180]$). Saturation thresholding avoids complex hue split logic for detection, while hue ranges are used for pen colour classification.
- **Black Pen Handling**: Black pen ink has low saturation similar to paper, but low lightness/value ($V \le \text{VAL\_MAX} = 200$).
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
- **Verdict**: **HSV combined segmentation won**. $S \ge 60$ provides an extremely sharp cut-off between white paper and coloured ink, and $V \le 200$ captures black pen ink without picking up paper background.

## Problems and how I solved them

1. **Leftover Table Borders inside Cell Crops**:
   - *Problem*: Even with cell insets, printed table rules surviving along cell boundaries were detected as dark ink.
   - *Solution*: `remove_table_lines` uses horizontal and vertical morphological opening kernels near cell margins and clear outer 2-pixel frames.

2. **Noise and Specks**:
   - *Problem*: Tiny dust or paper specks produced small connected components.
   - *Solution*: `drop_edge_blobs` and connected component area filtering (`MIN_BLOB_AREA = 12`) remove specks without eroding thin pen strokes.
