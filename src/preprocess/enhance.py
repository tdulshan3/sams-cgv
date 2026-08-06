"""Greyscale conversion, denoising, shadow removal and contrast enhancement.

Turns ``ctx["warped"]`` (a flattened colour photo of the sheet) into
``ctx["grey"]``: one channel, evenly lit, denoised. M4's thresholding lives or
dies on this output — see ``BUILD_SPEC.md`` section 9.3.
"""

from __future__ import annotations

import numpy as np


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
