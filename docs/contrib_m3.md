# M3: Greyscale & Enhancement

Individual contribution notes, per the coursework brief.

## What I built

`src/preprocess/enhance.py`, the stage that turns `ctx["warped"]` (a flattened
colour photo of the sheet) into `ctx["grey"]`; one channel, evenly lit,
denoised: for M4's thresholding.

- `to_grey`: four conversion methods (`average`, `luminosity`, `lightness`,
  `max_channel`). `average` and `luminosity` are hand-written in NumPy rather
  than `cv2.cvtColor`, so the channel weights are explicit.
- `denoise`: four methods (`gaussian`, `median`, `bilateral`, `nlmeans`),
  measured against each other with PSNR and runtime rather than picked by eye.
- `estimate_background` / `remove_shadow`: morphological background
  estimation (dilate + median blur) followed by division and full-range
  rescale, to flatten the shadow gradient every phone photo carries.
- `enhance_contrast`: global histogram equalisation vs CLAHE.
- `EnhanceStage`: chains grey → shadow removal → denoise → contrast, every
  step driven by `src/config.py`, `figures()` exposing all four intermediates
  to the step viewer.
- `tests/test_enhance.py` and `tools/make_m3_figures.py` (the five report
  figures listed in `BUILD_SPEC.md` §9.3).

## Techniques and libraries

- **Luminosity greyscale** (`0.299R + 0.587G + 0.114B`): the human eye's cone
  cells are most sensitive to green and least to blue, so a plain average
  under-weights how bright a pixel actually looks. This matters here because
  the ruled table lines and the blue pen ink sit at different points on that
  weighting, and getting it wrong changes how much contrast survives into
  `ctx["grey"]`.
- **Bilateral filtering**: weighs neighbouring pixels by both spatial
  distance and intensity difference, so it smooths flat paper texture without
  blurring across a strong edge such as a pen stroke boundary, unlike a
  gaussian blur, which treats every neighbour the same regardless of contrast.
- **Shadow removal by background division**: dilating with a kernel wider
  than any pen stroke erases the strokes and leaves the paper's own lighting
  behind; median blur then smooths that into a slowly-varying surface.
  Dividing the original by this surface cancels the shadow rather than just
  brightening the whole image, because a shadowed pixel and its local
  background are both dim, so their ratio stays close to what an unshadowed
  pixel's ratio would be.
- **CLAHE over global histogram equalisation**: a signing sheet is mostly
  blank paper. Global equalisation redistributes the *whole* image's
  histogram, so the huge blank-paper spike gets stretched hard and its noise
  is amplified along with it. CLAHE equalises small tiles independently, so a
  quiet tile stays quiet and the stretching only fires where there is
  something: ink, to bring out.

## Problems and how I solved them

- **`fastNlMeansDenoising`'s default filter strength barely denoises.**
  OpenCV's default `h=3` is tuned for very light noise; against the noise
  levels visible on the phone photos it left the image almost untouched,
  which the synthetic-noise unit test caught immediately (`result.var() <
  noisy.var()` failed for nlmeans and only nlmeans). Fixed by adding
  `NLMEANS_H = 10` to `config.py` and passing it through, the point of
  "no tunable number hard-coded" is exactly to make a fix like this a one-line
  config change instead of a scavenger hunt.
- **Shadow division alone does not reach the full 0–255 range.** Dividing by
  the estimated background flattens the *gradient*, but a shadowed corner's
  darkest ink still divides down to a mid-grey ratio, not black, so the
  output arrived compressed into a narrow band. Fixed by rescaling the result
  to the full range after the divide (`fix(preprocess): rescale to full range
  after shadow division`).
- **M2's `GeometryStage` returned near-empty crops in this environment** (a
  few tens of pixels on a side) rather than the flattened sheet, which would
  have made every M3 report figure a blank rectangle. Since `enhance.py`
  itself only reads `ctx["warped"]` and never calls the geometry code, the
  enhancement chain itself is unaffected, but `tools/make_m3_figures.py`
  needed a source image, so it uses the same crude fractional crop
  `tools/make_fixtures.py` already relies on instead. Worth flagging to M1/M2
; the fallback is a one-line swap back to `GeometryStage` once that is
  confirmed fixed.
- **CLAHE's default clip limit amplifies paper texture as well as ink.**
  Swept `CLAHE_CLIP` from 1.0 to 4.0 against Otsu binarisation on all five
  real sheets: every increase raised both the noise standard deviation in a
  blank patch and the ink percentage picked up over the whole page, with no
  corresponding gain once bilateral denoising has already run. Settled on
  `1.5` (down from the spec's starting suggestion of `2.0`).

## Evidence

- `outputs/figures/m3_greyscale_methods.png`; the four greyscale methods
  side by side on one sheet.
- `outputs/figures/m3_histograms.png`, pixel histogram before and after
  CLAHE on the same axes; the blank-paper spike does not get amplified the
  way it would under global equalisation.
- `outputs/figures/m3_denoise_comparison.png`, four denoise methods, full
  image and a zoomed signature crop; bilateral keeps stroke edges visibly
  sharper than gaussian or nlmeans at the same noise level.
- `outputs/figures/m3_denoise_metrics.png`, PSNR and runtime bar charts.
  Bilateral has the best PSNR of the four; nlmeans is both the slowest and,
  at its current settings, the weakest, an honest result, not the outcome I
  expected going in.
- `outputs/figures/m3_shadow_removal.png`, original, estimated background,
  and flattened, on the sheet with the most visible lighting gradient.
- `pytest tests/test_enhance.py -q`, 6 tests, covering the maths in §11 of
  `BUILD_SPEC.md` plus the PSNR/runtime measurement.
