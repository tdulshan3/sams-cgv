"""Draw M2's five report figures (BUILD_SPEC.md section 9.2).

    python tools/make_m2_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.io.image_loader import load_image, resize_to_width  # noqa: E402
from src.preprocess.deskew import (  # noqa: E402
    estimate_skew_angle,
    find_sheet_corners,
    four_point_warp,
)
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("m2figures")


def _rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR to RGB, and greyscale straight through."""
    return bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _panel(axis, image: np.ndarray, title: str) -> None:
    axis.imshow(_rgb(image), cmap="gray" if image.ndim == 2 else None)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _save(figure, name: str) -> Path:
    path = config.FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote %s", path.relative_to(config.ROOT))
    return path


def _stages(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """The intermediate images the geometry stage passes through."""
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blurred, config.CANNY_LOW, config.CANNY_HIGH)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outline = bgr.copy()
    if contours:
        cv2.drawContours(outline, [max(contours, key=cv2.contourArea)], -1, (0, 255, 0), 3)

    corners = find_sheet_corners(bgr)
    if corners is not None:
        warped = four_point_warp(bgr, corners)
    else:
        warped = _rotate(bgr, _clamped_skew(grey))
    return {"grey": grey, "edges": edges, "outline": outline, "warped": warped}


def _clamped_skew(grey: np.ndarray) -> float:
    angle = estimate_skew_angle(grey)
    return max(-config.MAX_SKEW_CORRECTION_DEG, min(config.MAX_SKEW_CORRECTION_DEG, angle))


def _rotate(bgr: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, growing the canvas so nothing is clipped."""
    h, w = bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - w / 2
    matrix[1, 2] += new_h / 2 - h / 2
    return cv2.warpAffine(bgr, matrix, (new_w, new_h))


def original_vs_warped(bgr: np.ndarray, name: str) -> None:
    steps = _stages(bgr)
    figure, axes = plt.subplots(1, 2, figsize=(9, 6), dpi=config.FIGURE_DPI)
    _panel(axes[0], bgr, f"original photo — {name}")
    _panel(axes[1], steps["warped"], "after geometry correction")
    figure.suptitle("M2 — geometry correction", fontsize=12)
    figure.tight_layout()
    _save(figure, "m2_original_vs_warped.png")


def corner_detection(bgr: np.ndarray) -> None:
    steps = _stages(bgr)
    corners = find_sheet_corners(bgr)
    overlay = bgr.copy()
    if corners is not None:
        cv2.polylines(overlay, [corners.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 4)
        for point in corners:
            cv2.circle(overlay, tuple(point.astype(int)), 12, (0, 0, 255), -1)
        caption = "4 corners found"
    else:
        caption = "no 4-sided outline above the area threshold\n(paper on a pale desk — rotation fallback used)"

    figure, axes = plt.subplots(1, 2, figsize=(9, 6), dpi=config.FIGURE_DPI)
    _panel(axes[0], steps["edges"], "Canny edge map")
    _panel(axes[1], overlay, caption)
    figure.suptitle("M2 — sheet outline detection", fontsize=12)
    figure.tight_layout()
    _save(figure, "m2_corner_detection.png")


def warp_steps(bgr: np.ndarray) -> None:
    steps = _stages(bgr)
    figure, axes = plt.subplots(1, 4, figsize=(15, 5), dpi=config.FIGURE_DPI)
    _panel(axes[0], bgr, "1. original")
    _panel(axes[1], steps["edges"], "2. edges")
    _panel(axes[2], steps["outline"], "3. largest contour")
    _panel(axes[3], steps["warped"], "4. corrected")
    figure.suptitle("M2 — the geometry stage, step by step", fontsize=12)
    figure.tight_layout()
    _save(figure, "m2_warp_steps.png")


def skew_correction(bgr: np.ndarray) -> None:
    """A deliberately tilted sheet, before and after angle correction."""
    tilted = _rotate(bgr, 6.0)
    grey = cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)
    measured = estimate_skew_angle(grey)
    corrected = _rotate(tilted, _clamped_skew(grey))

    figure, axes = plt.subplots(1, 2, figsize=(9, 6), dpi=config.FIGURE_DPI)
    _panel(axes[0], tilted, "tilted by 6.0° on purpose")
    _panel(axes[1], corrected, f"corrected — Hough measured {measured:+.2f}°")
    figure.suptitle("M2 — residual skew correction", fontsize=12)
    figure.tight_layout()
    _save(figure, "m2_skew_correction.png")


def all_sheets_grid(sheets: list[Path]) -> None:
    figure, axes = plt.subplots(1, len(sheets), figsize=(3.2 * len(sheets), 5), dpi=config.FIGURE_DPI)
    for axis, path in zip(np.atleast_1d(axes), sheets):
        bgr = resize_to_width(load_image(path))
        used = "warp" if find_sheet_corners(bgr) is not None else "rotate"
        _panel(axis, _stages(bgr)["warped"], f"{path.stem}\n({used})")
    figure.suptitle("M2 — all five sheets after geometry correction", fontsize=12)
    figure.tight_layout()
    _save(figure, "m2_all_sheets_grid.png")


def main() -> int:
    config.ensure_dirs()
    sheets = sorted(config.SHEETS.glob("*.png"))
    if not sheets:
        print(f"error: no sheets in {config.SHEETS}")
        return 2

    reference = resize_to_width(load_image(sheets[0]))
    original_vs_warped(reference, sheets[0].stem)
    corner_detection(reference)
    warp_steps(reference)
    skew_correction(reference)
    all_sheets_grid(sheets)

    print(f"\nM2 figures written to {config.FIGURES.relative_to(config.ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
