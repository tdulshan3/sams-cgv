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

import time

import cv2
import numpy as np
from skimage.filters import threshold_sauvola as _sk_threshold_sauvola

from src.config import ADAPTIVE_BLOCK, ADAPTIVE_C, SAUVOLA_WINDOW, THRESHOLD_GLOBAL_VALUE
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


def threshold_sauvola(grey: np.ndarray, window: int = SAUVOLA_WINDOW) -> np.ndarray:
    """Sauvola's local threshold, purpose-built for document images.

    Like :func:`threshold_adaptive`, the cut-off is local rather than global,
    but the formula also scales with the local *standard deviation*: a
    smooth patch of paper gets a threshold close to its own mean (so faint
    texture is not read as ink), while a patch with real contrast — an edge
    of a printed line or a pen stroke — gets a threshold pulled further from
    the mean. That extra term is what ``skimage`` was built around for
    scanned documents specifically, which is why it is worth comparing
    against plain adaptive thresholding here (T5).

    Args:
        grey: 2-D ``uint8`` greyscale image.
        window: Local neighbourhood size in pixels. Must be odd and at
            least 3.

    Returns:
        2-D ``uint8`` array, same shape as ``grey``, ink = 255.

    Raises:
        ValueError: ``window`` is even or smaller than 3.
    """
    if window < 3:
        raise ValueError(f"window must be at least 3, got {window}")
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")

    local_threshold = _sk_threshold_sauvola(grey.astype(np.float64), window_size=window)
    return np.where(grey.astype(np.float64) < local_threshold, 255, 0).astype(np.uint8)


def line_survival_ratio(binary: np.ndarray, min_len_ratio: float = 0.5) -> float:
    """Estimate whether long, unbroken horizontal runs of ink survive in
    ``binary`` — the printed table lines M5 has to find next.

    M5 has not landed yet, so there is no real line detector to ask "did
    this method keep your lines?" against. This is a stand-in that measures
    the same thing a wide horizontal morphological opening would find: it
    keeps only ink that forms a run at least ``min_len_ratio`` of the image
    width, then reports the widest surviving row as a fraction of the full
    width. A real table line should score close to 1.0; noise and short pen
    strokes score close to 0.0.

    Read this next to ``ink_percent`` — not alone. Heavy morphological
    closing can weld disconnected ink into a long run and inflate this
    number while also gluing a signature to the table border, which is
    exactly the failure mode T6 warns about.

    Args:
        binary: 2-D ``uint8`` image, ink = 255.
        min_len_ratio: Minimum run length, as a fraction of image width, to
            count as a candidate line.

    Returns:
        Fraction in ``[0, 1]``: widest surviving horizontal run over image
        width.
    """
    width = binary.shape[1]
    kernel_length = max(1, int(width * min_len_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    long_runs = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if not long_runs.any():
        return 0.0
    row_coverage = long_runs.sum(axis=1) / 255.0
    return float(row_coverage.max() / width)


def compare_methods(grey: np.ndarray) -> list[dict]:
    """Run all four thresholding methods on the same image and measure each.

    T5 asks for a measured comparison, not a claim about which one "looks
    best". Per method: ink percentage (should be a few percent of the page,
    not 40), connected component count (fewer specks is cleaner), the line
    survival estimate above, and wall-clock runtime.

    Args:
        grey: 2-D ``uint8`` greyscale image.

    Returns:
        One dict per method — keys ``method``, ``binary``, ``ink_percent``,
        ``components``, ``line_survival``, ``runtime_s`` — in the order
        global, otsu, adaptive, sauvola.
    """
    methods = {
        "global": lambda g: threshold_global(g),
        "otsu": lambda g: threshold_otsu(g)[0],
        "adaptive": lambda g: threshold_adaptive(g),
        "sauvola": lambda g: threshold_sauvola(g),
    }

    results = []
    for name, apply_method in methods.items():
        started = time.perf_counter()
        binary = apply_method(grey)
        elapsed = time.perf_counter() - started

        n_labels, _ = cv2.connectedComponents(binary)
        results.append(
            {
                "method": name,
                "binary": binary,
                "ink_percent": 100.0 * np.count_nonzero(binary) / binary.size,
                "components": n_labels - 1,  # label 0 is the background
                "line_survival": line_survival_ratio(binary),
                "runtime_s": elapsed,
            }
        )
    return results
