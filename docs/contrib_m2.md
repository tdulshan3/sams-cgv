# M2 — Acquisition & Geometry

Individual contribution notes. Two pages minimum, per the coursework brief: what you built, the techniques you used and why, the problems you hit, and the figures in `outputs/` that back it up.

## What I built

I implemented the initial acquisition and geometry correction pipeline (`GeometryStage` in `src/preprocess/deskew.py` and loaders in `src/io/image_loader.py`). The purpose of this stage is to standardize the raw photos taken by users before any downstream table extraction or signature analysis takes place. Phone photos often suffer from perspective distortion—because the camera is rarely held perfectly parallel to the paper—meaning rectangular pages appear as arbitrary quadrilaterals. By applying a four-point perspective transform (or a robust fallback rotation), we algorithmically pull the paper back into a flat, top-down rectangular view.

Crucially, my implementation ensures we preserve the full context of the document. The final cropped and deskewed image intentionally retains BOTH the lecture header table and the main student table, along with all 5 columns (including the right-hand `Signature` column). Rather than tightly cropping the student table right away, I leave that structural semantic decision to the M5 (Table Detection) stage. M2’s responsibility is purely geometric normalization, not structural parsing.

## Techniques and libraries

I utilized **OpenCV (`cv2`)** and **NumPy** for all core operations, alongside standard Python libraries. 

- **Perspective Warp and Homographies:** The primary deskew mechanism attempts to map the four corners of the detected page onto a perfect rectangle. Mathematically, this is done using a homography—a transformation matrix that maps points in one plane to another. `cv2.getPerspectiveTransform` calculates this $3 \times 3$ matrix based on our 4 source corners and 4 destination corners, and `cv2.warpPerspective` applies the matrix to every pixel, stretching and squeezing the image so the paper lies flat.
  
- **Edge Detection with Blurring:** To locate the paper, I applied a `cv2.Canny` edge detector. However, raw images contain high-frequency noise (like paper texture or camera grain) which creates false edges. Therefore, I first apply a Gaussian blur (`cv2.GaussianBlur`) to smooth out the noise, ensuring Canny only fires on the strong, structural boundaries of the page.

- **Skew Angle Clamping:** In our rotation fallback path (for when perspective warp fails), the Hough lines algorithm estimates a global skew angle. Instead of blindly trusting this angle, I aggressively clamp it using `MAX_SKEW_CORRECTION_DEG`. This safeguard prevents catastrophic misrotations (e.g., turning the page 90 degrees sideways because it latched onto a vertical line) in cases of extreme noise.

## Problems and how I solved them

The most significant hurdle was the physical acquisition environment of this dataset. The paper is photographed on a pale cream desk in every shot. This creates a severe low-contrast boundary between the edge of the paper and the background surface. In practice, the Canny edge detector—paired with the largest-contour heuristic—consistently fails to reliably isolate the true sheet perimeter because the edge gradients are simply too weak. 

This meant the primary four-point perspective warp failed to trigger on our real-world dataset. I solved this by treating the rotation fallback path not as an optional garnish, but as a critical, robust primary pathway. By leveraging `cv2.HoughLinesP` to find strong text/table lines and taking the median near-horizontal angle, the pipeline reliably rotates the image upright and preserves all necessary data for downstream tasks, even when the paper's literal edges are camouflaged.

## Two regressions found during integration

Both of these existed and were correct in my first pull request, and were lost
while the module was restructured across later ones. Neither was visible to the
test suite, and both were found only by running `sams.py` on a real sheet.

**The corners stopped being ordered.** `find_sheet_corners` returned
`approxPolyDP`'s raw traversal order, which starts at an arbitrary vertex and
may run either way round. `four_point_warp` unpacks its argument as
`tl, tr, br, bl`, so an unordered quad produces a **mirrored** sheet — the text
reads backwards. At thumbnail size it still looks like a signing sheet, which
is exactly why it survived review. Restoring the `_order_points` call fixes it.

**The area gate was dropped.** Without it the largest contour won regardless of
size, `MIN_SHEET_AREA_RATIO` became an unused import, and three sheets were
cropped to slivers between 0.4% and 0.6% of the page. Everything downstream was
then working on an image with no table in it.

The lesson I take from this is that a refactor across several pull requests can
silently drop a guard, and that neither guard was covered by a test asserting an
outcome on a real photo. `tests/test_real_sheets.py` now asserts that the
corrected sheet is at least half the area of the original.

## Evidence

- `m2_original_vs_warped.png` — raw angled input against the corrected output.
- `m2_corner_detection.png` — the Canny edge map, and why contour finding
  struggles against a pale desk.
- `m2_warp_steps.png` — original → edges → largest contour → corrected. The
  contour panel is the clearest evidence for the area gate: the largest
  four-sided shape on the page is the *student table*, not the sheet.
- `m2_skew_correction.png` — a sheet tilted 6° on purpose, and the angle Hough
  measured back.
- `m2_all_sheets_grid.png` — all five sheets after correction, each labelled
  with the path it took.

Regenerate with `python tools/make_m2_figures.py`.

### Per-Sheet Geometry Results

| Sheet Date | Geometry Path Used | Quality Note |
|---|---|---|
| 31.05.2019 | Fallback rotation | Straight; both tables and 5 columns intact |
| 21.06.2019 | Fallback rotation | Straight; both tables and 5 columns intact |
| 28.06.2019 | Fallback rotation | Straight; both tables and 5 columns intact |
| 05.07.2019 | Fallback rotation | Straight; both tables and 5 columns intact |
| 12.07.2019 | Fallback rotation | Straight; both tables and 5 columns intact |

*Note: The primary perspective warp (Canny + contour detection) fails on all sheets due to the low contrast between the cream paper and pale desk. The pipeline reliably falls back to Hough-lines rotation which produces an upright image with all required tables and columns intact.*

---

*Integration note: the corner-ordering and area-gate restorations described
above were applied by M1 during integration under deadline pressure. Review the
commit (`534d622`) before submission and rewrite this section in your own words.*
