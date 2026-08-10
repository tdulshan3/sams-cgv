"""Generate the five M3 report figures into ``outputs/figures/``.

Run once M2's real geometry stage is in place, since these figures compare
enhancement methods on an actual warped sheet rather than a fixture crop:

    python tools/make_m3_figures.py

Written by M3 for BUILD_SPEC.md section 9.3. Not imported by ``src/``, a
report-asset script, not part of the pipeline.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.preprocess.enhance import (  # noqa: E402
    denoise,
    enhance_contrast,
    estimate_background,
    remove_shadow,
    to_grey,
)
from src.utils.logging import get_logger  # noqa: E402
from tools.make_fixtures import SHEET_CROP, _crop  # noqa: E402

log = get_logger("m3_figures")

GREY_METHODS = ["average", "luminosity", "lightness", "max_channel"]
DENOISE_METHODS = ["gaussian", "median", "bilateral", "nlmeans"]

# Worst-shadow sheet by eye, used for the shadow-removal and histogram
# figures where the effect needs to be visible.
SHADOW_SHEET = "31.05.2019.png"


def _warped(sheet_name: str) -> np.ndarray:
    """A flattened sheet to build figures from.

    Uses the same crude fractional crop as ``tools/make_fixtures.py`` rather
    than the real ``GeometryStage``; at the time this was written M2's
    corner detection was returning near-empty crops on this machine, which
    would have made every M3 figure a blank rectangle. Swap this back to
    ``GeometryStage`` once that is confirmed fixed.
    """
    bgr = cv2.imread(str(config.SHEETS / sheet_name), cv2.IMREAD_COLOR)
    return _crop(bgr, SHEET_CROP)


def _add_noise(image: np.ndarray, sigma: float = 20.0, seed: int = 0) -> np.ndarray:
    """Synthetic gaussian noise, for the PSNR/SSIM denoise comparison."""
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _save(fig: plt.Figure, name: str) -> None:
    config.ensure_dirs()
    path = config.FIGURES / name
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.relative_to(config.ROOT))


def _panel_height(width: float, image: np.ndarray, n_cols: int) -> float:
    """Figure height that keeps each panel close to the source image's
    aspect ratio, so a wide sheet crop does not leave tall blank margins."""
    aspect = image.shape[0] / image.shape[1]
    return max(3.0, (width / n_cols) * aspect)


def make_greyscale_methods_figure(bgr: np.ndarray) -> None:
    height = 2 * _panel_height(9, bgr, 2) + 0.6
    fig, axes = plt.subplots(2, 2, figsize=(9, height))
    for ax, method in zip(axes.flat, GREY_METHODS):
        ax.imshow(to_grey(bgr, method), cmap="gray", vmin=0, vmax=255)
        ax.set_title(method)
        ax.axis("off")
    fig.suptitle("M3, greyscale conversion methods")
    _save(fig, "m3_greyscale_methods.png")


def make_histograms_figure(grey: np.ndarray) -> None:
    equalised = enhance_contrast(grey, "clahe")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(grey.ravel(), bins=256, range=(0, 255), alpha=0.6, label="before CLAHE")
    ax.hist(equalised.ravel(), bins=256, range=(0, 255), alpha=0.6, label="after CLAHE")
    ax.set_xlabel("pixel intensity")
    ax.set_ylabel("pixel count")
    ax.set_title("M3, histogram before and after CLAHE")
    ax.legend()
    _save(fig, "m3_histograms.png")


def make_denoise_comparison_figure(grey: np.ndarray) -> tuple[list[str], list[float], list[float]]:
    noisy = _add_noise(grey)
    results = {m: denoise(noisy, m) for m in DENOISE_METHODS}

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    for col, method in enumerate(DENOISE_METHODS):
        axes[0, col].imshow(results[method], cmap="gray", vmin=0, vmax=255)
        axes[0, col].set_title(method)
        axes[0, col].axis("off")

        # Zoomed crop of one signature-sized region, to show edge sharpness.
        y0, x0, size = grey.shape[0] // 2, grey.shape[1] // 2, 120
        crop = results[method][y0 : y0 + size, x0 : x0 + size]
        axes[1, col].imshow(crop, cmap="gray", vmin=0, vmax=255)
        axes[1, col].set_title(f"{method} (zoom)")
        axes[1, col].axis("off")
    fig.suptitle("M3, denoise comparison, full image and zoomed crop")
    _save(fig, "m3_denoise_comparison.png")

    methods, psnrs, times = [], [], []
    for method in DENOISE_METHODS:
        start = time.perf_counter()
        result = denoise(noisy, method)
        elapsed = time.perf_counter() - start
        methods.append(method)
        psnrs.append(peak_signal_noise_ratio(grey, result))
        times.append(elapsed)
        log.info(
            "%-10s psnr=%.2f dB  ssim=%.3f  runtime=%.3f s",
            method,
            psnrs[-1],
            structural_similarity(grey, result),
            elapsed,
        )
    return methods, psnrs, times


def make_denoise_metrics_figure(methods: list[str], psnrs: list[float], times: list[float]) -> None:
    fig, (ax_psnr, ax_time) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_psnr.bar(methods, psnrs, color="tab:blue")
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title("Denoise quality")

    ax_time.bar(methods, times, color="tab:orange")
    ax_time.set_ylabel("seconds")
    ax_time.set_title("Denoise runtime")
    fig.suptitle("M3, denoise metrics per method")
    _save(fig, "m3_denoise_metrics.png")


def make_shadow_removal_figure(grey: np.ndarray) -> None:
    background = estimate_background(grey)
    flattened = remove_shadow(grey)

    fig, axes = plt.subplots(1, 3, figsize=(13, _panel_height(13, grey, 3) + 0.6))
    for ax, image, title in zip(
        axes, [grey, background, flattened], ["original grey", "estimated background", "flattened"]
    ):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"M3, shadow removal ({SHADOW_SHEET})")
    _save(fig, "m3_shadow_removal.png")


def main() -> int:
    config.ensure_dirs()
    sample = _warped(sorted(p.name for p in config.SHEETS.glob("*.png"))[0])
    sample_grey = to_grey(sample, "luminosity")

    make_greyscale_methods_figure(sample)
    make_histograms_figure(sample_grey)
    methods, psnrs, times = make_denoise_comparison_figure(sample_grey)
    make_denoise_metrics_figure(methods, psnrs, times)

    shadow_grey = to_grey(_warped(SHADOW_SHEET), "luminosity")
    make_shadow_removal_figure(shadow_grey)

    print("Wrote 5 figures to outputs/figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
