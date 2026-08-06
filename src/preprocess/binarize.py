"""Thresholding and morphology.

Turns ``ctx["grey"]`` (M3's evenly-lit, denoised greyscale sheet) into
``ctx["binary"]``: a strictly two-valued image where ink is 255 (white) and
paper is 0 (black). M5's line detection reads this directly, so a broken
table line or a signature welded to its border here becomes M5's problem
two stages later — see ``BUILD_SPEC.md`` section 9.4.

**Convention that must never change: ink = 255, paper = 0.**
``cv2.THRESH_BINARY_INV`` gives that polarity directly; every function below
keeps it.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import ADAPTIVE_BLOCK, ADAPTIVE_C, THRESHOLD_GLOBAL_VALUE
from src.utils.stage import Stage


def threshold_global(grey: np.ndarray, value: int = THRESHOLD_GLOBAL_VALUE) -> np.ndarray:
    """Fixed threshold baseline: ink = 255, paper = 0.

    Kept deliberately simple and deliberately wrong. One number cannot suit
    a photo whose corner sits in shadow — a value tuned for the lit half of
    the page buries the shadowed half in false ink, or the reverse. It is
    kept as a report figure (``m4_global_failure.png``) precisely because it
    fails, not despite it.

    Args:
        grey: 2-D ``uint8`` greyscale image.
        value: Pixels darker than this become ink.

    Returns:
        2-D ``uint8`` array, same shape as ``grey``, values in ``{0, 255}``.
    """
    _, binary = cv2.threshold(grey, value, 255, cv2.THRESH_BINARY_INV)
    return binary


def otsu_between_class_variance(grey: np.ndarray) -> np.ndarray:
    """Between-class variance at every candidate threshold, Otsu's own search
    written out by hand rather than delegated to ``cv2.THRESH_OTSU``.

    The idea: treat every level ``t`` as a hypothetical cut between "paper"
    and "ink" pixels. A good cut is one where the two resulting classes are
    each tight around their own mean and far apart from each other — that
    separation is exactly what between-class variance measures, and it is
    equivalent to minimising the variance *within* each class, which is
    Otsu's original formulation.

    For every ``t`` in 0..255:
        ``w0, w1``  — fraction of pixels below / at-or-above ``t``
        ``m0, m1``  — mean intensity of each class
        ``variance = w0 * w1 * (m0 - m1) ** 2``

    Computed with cumulative sums rather than a 256-iteration Python loop —
    same maths, vectorised.

    Args:
        grey: 2-D ``uint8`` greyscale image.

    Returns:
        ``float64`` array of length 256: the variance at each threshold.
    """
    histogram, _ = np.histogram(grey, bins=256, range=(0, 256))
    total_pixels = histogram.sum()
    if total_pixels == 0:
        return np.zeros(256, dtype=np.float64)

    probabilities = histogram.astype(np.float64) / total_pixels
    levels = np.arange(256, dtype=np.float64)

    weight0 = np.cumsum(probabilities)          # P(pixel <= t)
    weight1 = 1.0 - weight0                      # P(pixel > t)
    running_sum = np.cumsum(probabilities * levels)
    total_mean = running_sum[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        mean0 = np.where(weight0 > 0, running_sum / weight0, 0.0)
        mean1 = np.where(weight1 > 0, (total_mean - running_sum) / weight1, 0.0)
        variance = weight0 * weight1 * (mean0 - mean1) ** 2

    return np.nan_to_num(variance, nan=0.0, posinf=0.0, neginf=0.0)


def threshold_otsu(grey: np.ndarray) -> tuple[np.ndarray, int]:
    """Otsu's method: the threshold that maximises between-class variance.

    Args:
        grey: 2-D ``uint8`` greyscale image.

    Returns:
        ``(binary, threshold)`` — the binary image (ink = 255) and the
        chosen threshold level, 0-255. Tested to land within one level of
        ``cv2.threshold(..., cv2.THRESH_OTSU)`` (``tests/test_binarize.py``).
    """
    variance = otsu_between_class_variance(grey)
    threshold = int(np.argmax(variance))
    _, binary = cv2.threshold(grey, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary, threshold


def threshold_adaptive(
    grey: np.ndarray,
    method: str = "gaussian",
    block: int = ADAPTIVE_BLOCK,
    c: int = ADAPTIVE_C,
) -> np.ndarray:
    """Locally adaptive threshold: each pixel is compared against the mean
    (or gaussian-weighted mean) of its own ``block`` x ``block`` neighbourhood
    rather than one global cut-off.

    This is normally the winner on phone photos of paper, because a single
    global value cannot be right everywhere at once when lighting drifts
    across the page — a local window adapts as it slides.

    Args:
        grey: 2-D ``uint8`` greyscale image.
        method: ``"mean"`` | ``"gaussian"``. Gaussian weights neighbours
            closer to the centre pixel more heavily, mean weights them all
            the same.
        block: Neighbourhood size in pixels. Must be odd and at least 3.
        c: Constant subtracted from the local mean before comparing —
            raising it makes the cut stricter, so fewer paper pixels flip
            to ink.

    Returns:
        2-D ``uint8`` array, same shape as ``grey``, ink = 255.

    Raises:
        ValueError: ``block`` is even or smaller than 3, or ``method`` is
            not one of the two above.
    """
    if block < 3:
        raise ValueError(f"block must be at least 3, got {block}")
    if block % 2 == 0:
        raise ValueError(f"block must be odd, got {block}")

    if method == "mean":
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method == "gaussian":
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError(f"unknown adaptive method {method!r}")

    return cv2.adaptiveThreshold(
        grey, 255, adaptive_method, cv2.THRESH_BINARY_INV, block, c
    )
