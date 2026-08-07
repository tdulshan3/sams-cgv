"""Line detection for the printed student table.

Uses morphological filtering as the primary method — far more reliable than
Hough on printed ruled lines. Hough is kept as a secondary cross-check.

Primary flow:
  binary (ink=255) -> erode+dilate with a long kernel -> line mask
  -> sum along the perpendicular axis -> find peaks -> merge nearby peaks
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.signal import find_peaks

from src.config import H_KERNEL_RATIO, LINE_MERGE_TOL, V_KERNEL_RATIO
from src.utils.logging import get_logger

log = get_logger("table")


def line_mask(binary: np.ndarray, orientation: str) -> np.ndarray:
    """Return a mask that keeps only long horizontal or vertical runs of ink.

    Args:
        binary: uint8 image, ink=255, paper=0.
        orientation: ``"horizontal"`` or ``"vertical"``.

    Returns:
        uint8 mask, same size as ``binary``.
    """
    if orientation == "horizontal":
        length = max(1, int(binary.shape[1] * H_KERNEL_RATIO))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    elif orientation == "vertical":
        length = max(1, int(binary.shape[0] * V_KERNEL_RATIO))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    else:
        raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")

    # Erode removes anything shorter than the kernel, dilate restores thickness.
    eroded = cv2.erode(binary, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated


def _peaks_from_profile(profile: np.ndarray, min_distance: int = 5) -> list[int]:
    """Find peaks in a 1-D projection profile.

    Args:
        profile: Sum along one axis of a line mask.
        min_distance: Minimum samples between peaks.

    Returns:
        Sorted list of peak positions.
    """
    if profile.max() == 0:
        return []
    threshold = profile.max() * 0.3
    peaks, _ = find_peaks(profile, height=threshold, distance=min_distance)
    return sorted(peaks.tolist())


def _merge_nearby(positions: list[int], tol: int) -> list[int]:
    """Merge positions that are within ``tol`` pixels of each other.

    Groups close values and replaces each group with its integer mean.
    """
    if not positions:
        return []
    merged: list[int] = []
    group: list[int] = [positions[0]]
    for pos in positions[1:]:
        if pos - group[-1] <= tol:
            group.append(pos)
        else:
            merged.append(int(round(sum(group) / len(group))))
            group = [pos]
    merged.append(int(round(sum(group) / len(group))))
    return merged


def detect_horizontal_lines(binary: np.ndarray, min_len_ratio: float = 0.5) -> list[int]:
    """Return Y pixel positions of horizontal table lines.

    Args:
        binary: uint8 binary image, ink=255.
        min_len_ratio: Passed to ``line_mask`` indirectly via config. Kept in
            the signature to match the spec contract.

    Returns:
        Sorted list of Y positions.
    """
    mask = line_mask(binary, "horizontal")
    profile = mask.sum(axis=1).astype(float)
    peaks = _peaks_from_profile(profile, min_distance=5)
    merged = _merge_nearby(peaks, LINE_MERGE_TOL)
    log.debug("horizontal lines detected: %s", merged)
    return merged


def detect_vertical_lines(binary: np.ndarray, min_len_ratio: float = 0.5) -> list[int]:
    """Return X pixel positions of vertical table lines.

    Args:
        binary: uint8 binary image, ink=255.
        min_len_ratio: Kept to match the spec contract.

    Returns:
        Sorted list of X positions.
    """
    mask = line_mask(binary, "vertical")
    profile = mask.sum(axis=0).astype(float)
    peaks = _peaks_from_profile(profile, min_distance=5)
    merged = _merge_nearby(peaks, LINE_MERGE_TOL)
    log.debug("vertical lines detected: %s", merged)
    return merged


def detect_lines_hough(binary: np.ndarray) -> tuple[list[int], list[int]]:
    """Cross-check line positions using HoughLinesP.

    Only lines within 5 degrees of axis-aligned are kept.

    Returns:
        Tuple of (y_positions, x_positions) for horizontal and vertical lines.
    """
    angle_tol = np.deg2rad(5)
    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(binary.shape[1] * 0.3),
        maxLineGap=20,
    )
    h_ys: list[int] = []
    v_xs: list[int] = []
    if lines is not None:
        # OpenCV 4 returns (N, 1, 4) here and OpenCV 5 returns (N, 4). We pin
        # opencv-python 5, where lines[:, 0] is a column of ints and unpacking
        # it raises. Reshaping first works on both.
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            angle = abs(np.arctan2(y2 - y1, x2 - x1))
            if angle < angle_tol:
                h_ys.append((y1 + y2) // 2)
            elif abs(angle - np.pi / 2) < angle_tol:
                v_xs.append((x1 + x2) // 2)
    h_ys = _merge_nearby(sorted(h_ys), LINE_MERGE_TOL)
    v_xs = _merge_nearby(sorted(v_xs), LINE_MERGE_TOL)
    log.debug("hough horizontal lines: %s", h_ys)
    log.debug("hough vertical lines: %s", v_xs)
    return h_ys, v_xs
