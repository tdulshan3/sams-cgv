# M4: Binarisation & Morphology

Individual contribution notes.

## What I built

`src/preprocess/binarize.py`; the stage that turns M3's evenly lit greyscale
sheet into a two-valued image where **ink is 255 and paper is 0**, then cleans
it with morphology. Fifteen functions and one `Stage` subclass, plus
`tests/test_binarize.py` and `tools/make_m4_figures.py`.

Everything downstream depends on that polarity convention. M5 hunts for long
white runs to find the table rules and M6 counts white pixels inside a cell, so
inverting it silently breaks two modules at once rather than producing an
obviously wrong picture.

| Function | What it does |
|---|---|
| `threshold_global` | Fixed cut-off. Kept as a failure exhibit, not as a candidate. |
| `threshold_otsu` | Otsu's method, the search written by hand in NumPy. |
| `threshold_adaptive` | Mean and Gaussian, a different cut-off per block. |
| `threshold_sauvola` | scikit-image's local method, built for documents. |
| `morph_clean` | Opening to remove speckle, closing to repair strokes. |
| `skeletonize_ink` | One-pixel-wide strokes, used by M6 and M8. |
| `clean_signature_crop` | The same cleaning tuned for small crops, for M8. |

## Techniques, and why each was chosen

**Otsu written by hand.** Calling `cv2.THRESH_OTSU` and stopping would have
been a line of code and nothing to say about it. Implementing the search,
build the 256-bin normalised histogram, and for every candidate `t` compute the
two class weights and means and maximise the between-class variance
`w₀·w₁·(m₀−m₁)²`; means the report can state the equation and show the variance
curve. `tests/test_binarize.py` checks the hand-written value against OpenCV's
to within one grey level, which is both a test and the evidence that the
implementation is right.

**Adaptive thresholding wins, and it wins for a reason.** A single global
cut-off cannot serve a photo whose corner is in shadow: the threshold that
keeps ink in the bright half erases it in the dark half. Adaptive decides per
block, so a shadowed corner is judged against its own neighbourhood.

**Sauvola has a trap.** `skimage.filters.threshold_sauvola` infers its dynamic
range `r` from the array's dtype. Handed a float array it takes the limits as
`(-1, 1)`, so `r` becomes 1.0 instead of ~128, the local threshold lands around
1000 on 0–255 data, every pixel falls below it and the entire page comes back
as ink. `SAUVOLA_R = 128.0` is passed explicitly for that reason, and the
config block says so, because the failure looks like a bug in the image rather
than in the call.

## The parameter sweep

`ADAPTIVE_BLOCK = 41` and `ADAPTIVE_C = 12` were chosen by sweeping block sizes
15–51 against `c` 5–15 on all five sheets. The result is a broad plateau rather
than a peak: ink coverage moves only between 7.6% and 10.0% across the whole
grid: so the values were picked to sit away from the edges rather than to
chase a maximum:

* `c = 5` is a cliff, not a slope. Component count jumps from ~400 to ~1100 as
  paper texture starts crossing the threshold. Anything from 8 up is stable.
* Larger blocks preserve the thin printed table rules better, because a thin
  dark line contrasts more strongly against a wider bright neighbourhood. Below
  block 21 line survival drops, which is M5's problem before it is mine.

## The trade-off I actually hit

Closing repairs a broken pen stroke, and it also welds a signature to the table
border it is touching. The sweep shows it plainly:

| open / close | ink % | components |
|---|---|---|
| 2 / 3 | 8.78 | 381 |
| 2 / 5 | 8.96 | 287 |
| 3 / 7 | 9.38 | 174 |

Falling component count with *rising* ink is not cleaning; it is separate
objects merging into one. By kernel 7 more than half the components on the page
have joined a neighbour, and on a signing sheet the nearest neighbour of a
signature is the row rule beneath it. `MORPH_OPEN_K = 2, MORPH_CLOSE_K = 3`
keeps the strokes and the rules apart, which is what M5 and M6 need even though
a larger kernel produces a prettier image.

## Evidence

* `m4_threshold_comparison.png`, global, Otsu, adaptive and Sauvola side by side
* `m4_otsu_histogram.png`, the grey histogram with the chosen threshold and the between-class variance curve
* `m4_global_failure.png`, the shadowed sheet where one global value cannot work
* `m4_morphology.png`, raw binary, after opening, after closing, with a zoomed crop
* `m4_metrics.png`, ink %, component count and runtime per method

`pytest tests/test_binarize.py` covers the polarity, the Otsu agreement, and
the effect of each morphology operation.

---

*Integration note: this document was drafted by M1 during integration from the
measurements recorded in `src/config.py` and the module's own code. Review it
and rewrite in your own words before submission.*
