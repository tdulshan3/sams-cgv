"""OpenCV calls whose return shape changed between major versions.

We pin ``opencv-python`` 5. Several OpenCV functions that returned an
``(N, 1, K)`` array in version 4 return ``(N, K)`` in version 5, and almost
every tutorial, Stack Overflow answer and textbook example on the internet was
written against version 4. The result is code that looks right, passes a
synthetic unit test, and dies on the first real image::

    for x1, y1, x2, y2 in lines[:, 0]:
        TypeError: cannot unpack non-iterable numpy.int32 object

This has already cost the group two modules — M2 in PR #3 and M5 in PR #11,
found only by running the pipeline on a real sheet. Rather than fix it a third
time, call the wrappers here and never call the underlying function directly.
``tests/test_pipeline.py`` fails the build if anyone does.
"""

from __future__ import annotations

import cv2
import numpy as np

EMPTY_SEGMENTS = np.empty((0, 4), dtype=np.int32)
"""What :func:`hough_line_segments` returns when nothing is found.

An empty array rather than ``None``, so callers can iterate unconditionally
instead of each inventing their own ``if lines is None`` guard.
"""


def hough_line_segments(
    binary: np.ndarray,
    *,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    threshold: int = 80,
    min_line_length: int = 50,
    max_line_gap: int = 10,
) -> np.ndarray:
    """Probabilistic Hough transform, with a return shape that does not move.

    Args:
        binary: Single channel image, ink 255 on paper 0.
        rho: Distance resolution in pixels.
        theta: Angle resolution in radians.
        threshold: Minimum votes for a line to be returned.
        min_line_length: Segments shorter than this are discarded.
        max_line_gap: Largest gap between points still treated as one line.

    Returns:
        An ``(N, 4)`` array of ``x1, y1, x2, y2``, on every OpenCV version.
        Empty when no lines are found — never ``None``.

    Example:
        >>> for x1, y1, x2, y2 in hough_line_segments(binary):
        ...     ...
    """
    lines = cv2.HoughLinesP(
        binary,
        rho,
        theta,
        threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None or lines.size == 0:
        return EMPTY_SEGMENTS
    # (N, 1, 4) on OpenCV 4, (N, 4) on OpenCV 5. Both flatten to (N, 4).
    return np.asarray(lines).reshape(-1, 4)
