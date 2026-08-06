"""Greyscale conversion, denoising, shadow removal and contrast enhancement.

Turns ``ctx["warped"]`` (a flattened colour photo of the sheet) into
``ctx["grey"]``: one channel, evenly lit, denoised. M4's thresholding lives or
dies on this output — see ``BUILD_SPEC.md`` section 9.3.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP,
    CLAHE_GRID,
    GAUSSIAN_KSIZE,
    MEDIAN_KSIZE,
    SHADOW_KERNEL,
)


def to_grey(bgr: np.ndarray, method: str = "luminosity") -> np.ndarray:
    """Collapse a BGR image to one channel.

    ``average`` and ``luminosity`` are written by hand with NumPy rather than
    ``cv2.cvtColor`` so the weights are visible. The luminosity weights
    (0.299 R, 0.587 G, 0.114 B) are not arbitrary — they come from how the
    human retina responds to colour. Cone cells are most numerous and most
    sensitive in the green part of the spectrum, so a green pixel *looks*
    brighter than a red or blue pixel of the same raw intensity even though a
    plain average treats all three the same. Weighting green highest and blue
    lowest is what makes the greyscale conversion match human perception
    instead of just measuring photon count.

    Args:
        bgr: Colour image, uint8, any number of rows/columns.
        method: ``"average"`` | ``"luminosity"`` | ``"lightness"`` |
            ``"max_channel"``.

    Returns:
        2-D ``uint8`` array, same height and width as ``bgr``.

    Raises:
        ValueError: ``method`` is not one of the four above.
    """
    b = bgr[..., 0].astype(np.float64)
    g = bgr[..., 1].astype(np.float64)
    r = bgr[..., 2].astype(np.float64)

    if method == "average":
        # Simple mean of the three channels. Treats every channel as equally
        # important, which is not how the human eye works — see luminosity.
        grey = (r + g + b) / 3.0
    elif method == "luminosity":
        # Weighted by how sensitive the human eye is to each colour: most
        # sensitive to green, least to blue. Same weights as ITU-R BT.601.
        grey = 0.299 * r + 0.587 * g + 0.114 * b
    elif method == "lightness":
        # Midpoint of the brightest and dimmest channel. Ignores the middle
        # channel entirely, so it is the least representative of the four.
        grey = (np.maximum.reduce([r, g, b]) + np.minimum.reduce([r, g, b])) / 2.0
    elif method == "max_channel":
        # Whichever channel is brightest at that pixel. Cheap, and biased
        # toward whatever the most saturated colour in the shot happens to be.
        grey = np.maximum.reduce([r, g, b])
    else:
        raise ValueError(f"unknown greyscale method {method!r}")

    return np.clip(grey, 0, 255).astype(np.uint8)


def denoise(grey: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Suppress sensor and paper-texture noise ahead of thresholding.

    Args:
        grey: 2-D ``uint8`` greyscale image.
        method: ``"gaussian"`` | ``"median"`` | ``"bilateral"`` | ``"nlmeans"``.

    Returns:
        2-D ``uint8`` array, same shape as ``grey``.

    Raises:
        ValueError: ``method`` is not one of the four above.
    """
    if method == "gaussian":
        # Blurs everything uniformly, edges included — fine for the sensor's
        # gaussian noise floor, but it also softens pen strokes.
        return cv2.GaussianBlur(grey, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0)
    if method == "median":
        # Replaces each pixel with the median of its neighbourhood. Removes
        # salt-and-pepper style outliers that a gaussian blur only smears.
        return cv2.medianBlur(grey, MEDIAN_KSIZE)
    if method == "bilateral":
        # Weighs neighbours by both spatial distance and intensity
        # difference, so it smooths flat paper texture without blurring
        # across a strong edge such as a pen stroke boundary.
        return cv2.bilateralFilter(
            grey, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
        )
    if method == "nlmeans":
        # Averages every pixel against similar-looking patches across the
        # whole image rather than just its local neighbourhood. Strong noise
        # removal, but the slowest of the four by a wide margin.
        return cv2.fastNlMeansDenoising(grey)
    raise ValueError(f"unknown denoise method {method!r}")


def estimate_background(grey: np.ndarray) -> np.ndarray:
    """Estimate the sheet's lighting map: what the page would look like with
    no ink on it.

    Dilating with a kernel wider than any pen stroke erases the strokes,
    leaving only the paper. A median blur then smooths what is left into a
    slowly-varying lighting surface — bright where a shadow does not fall,
    dim in the corner where it does.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (SHADOW_KERNEL, SHADOW_KERNEL)
    )
    dilated = cv2.dilate(grey, kernel)
    ksize = SHADOW_KERNEL if SHADOW_KERNEL % 2 == 1 else SHADOW_KERNEL + 1
    return cv2.medianBlur(dilated, ksize)


def remove_shadow(grey: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting so a corner shadow does not skew thresholding.

    Dividing the page by its own lighting map cancels the shadow out: a pixel
    that is dim only because it sits in shadow reads close to 1.0 (its own
    brightness over the background's, which is dim there too), while ink
    stays dark relative to the paper around it either way.
    """
    background = estimate_background(grey)
    ratio = grey.astype(np.float64) / (background.astype(np.float64) + 1e-6)
    flattened = np.clip(ratio * 255.0, 0, 255)

    # The divide alone rarely reaches 0 or 255 — a shadowed corner's darkest
    # ink still divides down to a mid-grey ratio, not black. Stretch the
    # result back out to the full range so downstream contrast and
    # thresholding get the dynamic range they expect.
    lo, hi = flattened.min(), flattened.max()
    if hi > lo:
        flattened = (flattened - lo) * (255.0 / (hi - lo))
    return flattened.astype(np.uint8)


def enhance_contrast(grey: np.ndarray, method: str = "clahe") -> np.ndarray:
    """Stretch contrast so faint pen strokes stand out from the paper.

    Args:
        grey: 2-D ``uint8`` greyscale image.
        method: ``"none"`` | ``"histeq"`` | ``"clahe"``.

    Returns:
        2-D ``uint8`` array, same shape as ``grey``.

    Raises:
        ValueError: ``method`` is not one of the three above.
    """
    if method == "none":
        return grey
    if method == "histeq":
        # Redistributes the whole image's histogram to be flat. A page that
        # is mostly blank paper has one huge histogram spike, so a global
        # equalisation stretches that spike hard and amplifies noise in it.
        return cv2.equalizeHist(grey)
    if method == "clahe":
        # Equalises each small tile of the page on its own, then blends tile
        # borders. A blank tile has little to stretch, so it stays quiet
        # instead of amplifying its own noise the way global equalisation
        # does — the local strokes are what get boosted.
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        return clahe.apply(grey)
    raise ValueError(f"unknown contrast method {method!r}")
