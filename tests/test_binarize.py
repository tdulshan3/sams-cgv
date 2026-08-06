"""Tests for thresholding and morphology.

Covers BUILD_SPEC.md section 11 (M4): two-valued output, ink polarity,
custom Otsu accuracy against OpenCV, the effect of opening and closing, and
parameter validation. Synthetic images only, per section 11's note that a
test needing a full sheet photo is slow and tells you less.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np
import pytest

from src.preprocess.binarize import (
    BinarizeStage,
    apply_threshold,
    clean_signature_crop,
    compare_methods,
    morph_clean,
    morph_close,
    morph_open,
    skeletonize_ink,
    structuring_element,
    threshold_adaptive,
    threshold_global,
    threshold_otsu,
    threshold_sauvola,
)

METHODS = ["global", "otsu", "adaptive", "sauvola"]


def _two_peak_image() -> np.ndarray:
    """A clean synthetic image with two well-separated intensity peaks —
    dark 'ink' pixels around 40, bright 'paper' pixels around 200 — so the
    correct threshold is unambiguous and OpenCV's own Otsu is a fair judge."""
    rng = np.random.default_rng(0)
    paper = rng.normal(200, 8, (100, 100))
    ink = rng.normal(40, 8, (100, 100))
    image = np.concatenate([paper, ink], axis=1)
    return np.clip(image, 0, 255).astype(np.uint8)


def _stroke_on_paper() -> np.ndarray:
    """White paper (near 255) with one dark pen stroke through it."""
    grey = np.full((60, 120), 235, dtype=np.uint8)
    grey[25:33, 10:110] = 30
    return grey


# -- polarity and output contract -----------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_output_is_strictly_two_valued(method):
    """Every method returns an image containing only 0 and 255 — the
    contract M5, M6 and M7 all read."""
    out = apply_threshold(_stroke_on_paper(), method)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 255}


@pytest.mark.parametrize("method", METHODS)
def test_ink_is_255_and_paper_is_0(method):
    """A dark stroke on white paper comes out white (255), and the paper
    around it comes out black (0). Getting this backwards silently breaks
    three downstream modules at once."""
    grey = _stroke_on_paper()
    out = apply_threshold(grey, method)

    assert out[29, 60] == 255, f"{method}: stroke centre should be ink (255)"
    assert out[5, 5] == 0, f"{method}: paper corner should be background (0)"


def test_binarize_stage_writes_two_valued_binary_to_ctx():
    """BinarizeStage honours the section 6.3 context contract: reads
    ctx['grey'], writes a two-valued ctx['binary'] with ink at 255."""
    stage = BinarizeStage()
    ctx = stage.run({"grey": _stroke_on_paper()})

    binary = ctx["binary"]
    assert set(np.unique(binary).tolist()) <= {0, 255}
    assert binary[29, 60] == 255
    assert set(stage.figures()) == {"binary raw", "binary cleaned"}


# -- Otsu ------------------------------------------------------------------


def test_custom_otsu_matches_opencv_within_one_level():
    """Hand-written Otsu on a clean two-peak image lands between the peaks
    and matches OpenCV's cv2.THRESH_OTSU within +/- 1 level."""
    grey = _two_peak_image()

    _, ours = threshold_otsu(grey)
    opencv_threshold, _ = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    assert 40 < ours < 200
    assert abs(ours - int(opencv_threshold)) <= 1


def test_otsu_on_a_flat_image_does_not_crash():
    """A blank image has no two classes to separate. Otsu must still return
    a usable threshold rather than dividing by a zero class weight."""
    binary, threshold = threshold_otsu(np.full((20, 20), 128, dtype=np.uint8))
    assert 0 <= threshold <= 255
    assert set(np.unique(binary).tolist()) <= {0, 255}


def test_sauvola_ink_coverage_stays_sane_on_a_mostly_blank_page():
    """Regression: Sauvola's dynamic range must be passed explicitly.

    Inferred from a float array's dtype limits it becomes 1.0 instead of
    ~128, the local threshold lands near 1000 on 0-255 data, and every
    pixel falls below it — a blank page comes out 97% ink. A signing sheet
    is a few percent ink, so anything near total coverage is the bug back.
    """
    page = np.full((200, 200), 230, dtype=np.uint8)
    page[90:100, 20:180] = 40                 # a single stroke

    binary = threshold_sauvola(page)
    ink_percent = 100.0 * np.count_nonzero(binary) / binary.size

    assert ink_percent < 20.0, f"sauvola marked {ink_percent:.0f}% of a blank page as ink"
    assert binary[95, 100] == 255, "the stroke itself should still be ink"


# -- morphology ------------------------------------------------------------


def test_opening_removes_isolated_single_pixels():
    """Opening deletes a lone speck while leaving a real stroke behind."""
    image = np.zeros((40, 40), dtype=np.uint8)
    image[5, 5] = 255                 # isolated speck
    image[20:24, 10:30] = 255         # a real stroke

    opened = morph_open(image, kernel_size=3)

    assert opened[5, 5] == 0, "isolated speck should be removed"
    assert opened[22, 20] == 255, "real stroke should survive"


def test_closing_fills_a_one_pixel_gap_in_a_stroke():
    """Closing bridges a hairline break, which is what a pen skip looks
    like after thresholding.

    The stroke here is 3 pixels thick, the thickness a real ballpoint
    leaves at the resolution the pipeline works at. See the test below for
    why thickness matters to which kernel can do this.
    """
    stroke = np.zeros((40, 40), dtype=np.uint8)
    stroke[19:22, 5:35] = 255
    stroke[19:22, 20] = 0             # a one pixel wide gap

    closed = morph_close(stroke, kernel_size=3)

    assert closed[20, 20] == 255, "the gap should be bridged"


def test_only_a_rectangular_kernel_bridges_a_gap_in_a_hairline():
    """At size 3, OpenCV's ellipse kernel is identical to a cross — it has
    no diagonal support. Closing a *one pixel thick* line with it therefore
    cannot bridge a gap: the dilation fills it, but the erosion immediately
    reopens it because the pixel above and below the gap were never set.
    Only the rectangle, which does include the diagonals, survives erosion.

    This is why MORPH_KERNEL_SHAPE is a config value and not a constant,
    and it is the reason the default stays 'ellipse': the gentler kernel is
    what keeps a signature from being welded onto the printed table line
    beneath it, and real strokes are thick enough not to need the rectangle.
    """
    hairline = np.zeros((40, 40), dtype=np.uint8)
    hairline[20, 5:35] = 255
    hairline[20, 20] = 0

    assert morph_close(hairline, 3, "rect")[20, 20] == 255
    assert morph_close(hairline, 3, "ellipse")[20, 20] == 0


def test_morph_clean_preserves_two_valued_output():
    """The full clean-up chain must not introduce intermediate greys."""
    rng = np.random.default_rng(3)
    noisy = (rng.random((50, 50)) > 0.5).astype(np.uint8) * 255
    assert set(np.unique(morph_clean(noisy)).tolist()) <= {0, 255}


def test_morph_kernels_of_size_one_are_a_no_op():
    """A kernel of 1 cannot change anything, so it returns the input rather
    than paying for a pointless OpenCV call."""
    image = np.zeros((10, 10), dtype=np.uint8)
    image[5, 5] = 255
    assert np.array_equal(morph_open(image, 1), image)
    assert np.array_equal(morph_close(image, 1), image)


def test_skeletonize_thins_strokes_and_keeps_polarity():
    """Skeletonisation reduces a thick bar to roughly its centre line, and
    ink stays 255."""
    bar = np.zeros((21, 60), dtype=np.uint8)
    bar[8:13, 5:55] = 255

    skeleton = skeletonize_ink(bar)

    assert set(np.unique(skeleton).tolist()) <= {0, 255}
    assert 0 < np.count_nonzero(skeleton) < np.count_nonzero(bar) / 2


def test_clean_signature_crop_returns_ink_255():
    """M8's small-crop helper keeps the project's polarity and can return
    either the filled mask or its skeleton."""
    mask = np.zeros((40, 80), dtype=np.uint8)
    mask[18:23, 10:70] = 255

    cleaned = clean_signature_crop(mask)
    skeleton = clean_signature_crop(mask, skeleton=True)

    assert set(np.unique(cleaned).tolist()) <= {0, 255}
    assert np.count_nonzero(cleaned) > 0
    assert np.count_nonzero(skeleton) < np.count_nonzero(cleaned)


# -- parameter validation --------------------------------------------------


def test_even_adaptive_block_raises_value_error():
    """An even block size raises a clear ValueError rather than letting
    OpenCV fail with an opaque assertion deep in native code."""
    grey = np.zeros((30, 30), dtype=np.uint8)

    with pytest.raises(ValueError, match="odd"):
        threshold_adaptive(grey, block=34)
    with pytest.raises(ValueError, match="at least 3"):
        threshold_adaptive(grey, block=1)


def test_even_sauvola_window_raises_value_error():
    """Sauvola validates its window the same way, for the same reason."""
    grey = np.zeros((30, 30), dtype=np.uint8)

    with pytest.raises(ValueError, match="odd"):
        threshold_sauvola(grey, window=24)


def test_unknown_method_names_raise_value_error():
    """No method selector silently falls through to a default — a typo in
    config.py should stop the run, not quietly change the algorithm."""
    grey = np.zeros((30, 30), dtype=np.uint8)

    with pytest.raises(ValueError):
        apply_threshold(grey, "not-a-method")
    with pytest.raises(ValueError):
        threshold_adaptive(grey, method="not-a-method")
    with pytest.raises(ValueError):
        structuring_element("diamond", 3)


# -- comparison harness ----------------------------------------------------


def test_compare_methods_reports_every_metric_for_every_method():
    """The T5 harness measures all four methods and reports the metrics the
    report tabulates, rather than asserting one method wins."""
    results = compare_methods(_stroke_on_paper())

    assert [r["method"] for r in results] == METHODS
    for result in results:
        assert 0.0 <= result["ink_percent"] <= 100.0
        assert result["components"] >= 0
        assert 0.0 <= result["line_survival"] <= 1.0
        assert result["runtime_s"] >= 0.0


def test_global_threshold_uses_its_configured_default():
    """threshold_global reads its cut-off from config rather than hard
    coding one, so the failure exhibit is reproducible from the config."""
    grey = _stroke_on_paper()
    assert np.array_equal(threshold_global(grey), threshold_global(grey, 127))
