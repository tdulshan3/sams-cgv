"""Tests for thresholding and morphology.

Covers BUILD_SPEC.md section 11 (M4): polarity, custom Otsu accuracy against
OpenCV, and the effect of opening/closing. Synthetic images only, per section
11's note that a test needing a full sheet photo is slow and tells you less.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import cv2
import numpy as np

from src.preprocess.binarize import threshold_otsu


def _two_peak_image() -> np.ndarray:
    """A clean synthetic image with two well-separated intensity peaks —
    dark 'ink' pixels around 40, bright 'paper' pixels around 200 — so the
    correct threshold is unambiguous and OpenCV's own Otsu is a fair judge."""
    rng = np.random.default_rng(0)
    paper = rng.normal(200, 8, (100, 100))
    ink = rng.normal(40, 8, (100, 100))
    image = np.concatenate([paper, ink], axis=1)
    return np.clip(image, 0, 255).astype(np.uint8)


def test_custom_otsu_matches_opencv_within_one_level():
    """Hand-written Otsu on a clean two-peak image lands between the peaks
    and matches OpenCV's cv2.THRESH_OTSU within +/- 1 level."""
    grey = _two_peak_image()

    _, ours = threshold_otsu(grey)
    opencv_threshold, _ = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    assert 40 < ours < 200
    assert abs(ours - int(opencv_threshold)) <= 1
