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

from src.config import THRESHOLD_GLOBAL_VALUE
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
