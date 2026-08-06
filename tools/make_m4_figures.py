"""Generate the five M4 report figures into ``outputs/figures/``.

Usage::

    python tools/make_m4_figures.py

Written by M4 for BUILD_SPEC.md section 9.4. Not imported by ``src/`` — a
report-asset script, not part of the pipeline.

Like ``tools/make_m3_figures.py``, this reads the sheets through the same
crude fractional crop ``tools/make_fixtures.py`` uses rather than through
M2's ``GeometryStage``, which currently returns near-empty crops on this
machine (see ``docs/contrib_m3.md``). The thresholding being illustrated
does not depend on how the sheet was cropped; swap ``_warped`` back to
``GeometryStage`` once M2's corner detection is confirmed fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.preprocess.binarize import (  # noqa: E402
    compare_methods,
    morph_close,
    morph_open,
    otsu_between_class_variance,
    threshold_adaptive,
    threshold_global,
    threshold_otsu,
    threshold_sauvola,
)
from src.preprocess.enhance import (  # noqa: E402
    denoise,
    enhance_contrast,
    remove_shadow,
    to_grey,
)
from src.utils.logging import get_logger  # noqa: E402
from tools.make_fixtures import SHEET_CROP, _crop  # noqa: E402

log = get_logger("m4_figures")

SAMPLE_SHEET = "31.05.2019.png"
"""Sheet used for the method-comparison and morphology figures."""

GRADIENT_SHEET = "12.07.2019.png"
"""Most evenly lit crop, so the synthetic gradient added in the global
failure figure is demonstrably the only lighting variation present."""

SHADOW_RAMP = (0.45, 1.0)
"""Left-to-right brightness multiplier for the controlled shadow."""

ZOOM = (slice(430, 700), slice(1450, 2100))
"""A signature-bearing region of the sample sheet, for the morphology
close-ups. Whole-sheet panels are too small to show a 2-pixel kernel's
effect at report size."""


def _grey(sheet_name: str, enhanced: bool = True) -> np.ndarray:
    """One sheet as M4 receives it — M3's full enhancement chain applied."""
    bgr = cv2.imread(str(config.SHEETS / sheet_name), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"error: could not read {config.SHEETS / sheet_name}")
    grey = to_grey(_crop(bgr, SHEET_CROP), "luminosity")
    if not enhanced:
        return grey
    return enhance_contrast(denoise(remove_shadow(grey), "bilateral"), "clahe")


def _save(fig: plt.Figure, name: str, rect: tuple[float, float, float, float] | None = None) -> None:
    config.ensure_dirs()
    path = config.FIGURES / name
    # Panels of a wide sheet crop are far shorter than their axes box, so
    # without this the rows sit in a sea of white space at report size.
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.relative_to(config.ROOT))


def _show(ax, image: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def make_threshold_comparison(grey: np.ndarray) -> None:
    """2x2: global, Otsu, adaptive and Sauvola on the same sheet."""
    _, otsu_t = threshold_otsu(grey)
    panels = [
        (threshold_global(grey), f"global (t={config.THRESHOLD_GLOBAL_VALUE})"),
        (threshold_otsu(grey)[0], f"otsu (t={otsu_t}, found by hand)"),
        (
            threshold_adaptive(grey),
            f"adaptive gaussian ({config.ADAPTIVE_BLOCK}/{config.ADAPTIVE_C})",
        ),
        (threshold_sauvola(grey), f"sauvola (window {config.SAUVOLA_WINDOW})"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (image, title) in zip(axes.flat, panels):
        _show(ax, image, title)
    fig.suptitle(f"M4 — thresholding methods on {SAMPLE_SHEET} (ink = white)")
    _save(fig, "m4_threshold_comparison.png")


def make_otsu_histogram(grey: np.ndarray) -> None:
    """The grey histogram with the chosen threshold marked, above the
    between-class variance curve that chose it."""
    variance = otsu_between_class_variance(grey)
    threshold = int(np.argmax(variance))
    opencv_t, _ = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fig, (ax_hist, ax_var) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"hspace": 0.15}
    )

    ax_hist.hist(grey.ravel(), bins=256, range=(0, 255), color="tab:blue", alpha=0.75)
    ax_hist.axvline(threshold, color="tab:red", lw=2, label=f"chosen t = {threshold}")
    ax_hist.set_ylabel("pixel count")
    ax_hist.set_title(
        "Grey histogram — the tall right peak is blank paper, "
        "the low left tail is ink",
        fontsize=10,
    )
    ax_hist.legend()

    ax_var.plot(variance, color="tab:green", lw=1.6)
    ax_var.axvline(threshold, color="tab:red", lw=2)
    ax_var.plot([threshold], [variance[threshold]], "o", color="tab:red", ms=7)
    ax_var.annotate(
        f"maximum at t = {threshold}\n(OpenCV chose {int(opencv_t)})",
        xy=(threshold, variance[threshold]),
        xytext=(0.55, 0.55),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "tab:red"},
        fontsize=9,
    )
    ax_var.set_xlabel("threshold t")
    ax_var.set_ylabel(r"$w_0 w_1 (m_0 - m_1)^2$")
    ax_var.set_title(
        "Between-class variance at every candidate threshold — "
        "Otsu picks the peak",
        fontsize=10,
    )
    ax_var.set_xlim(0, 255)

    fig.suptitle(f"M4 — Otsu's threshold search, computed by hand ({SAMPLE_SHEET})")
    _save(fig, "m4_otsu_histogram.png")


def make_global_failure() -> None:
    """Why one global number cannot survive uneven lighting.

    A controlled demonstration: an evenly lit sheet with a known linear
    shadow applied, so the gradient is the only variable. Otsu is included
    deliberately — it is the *optimal* global threshold and it fails in
    exactly the same way, which is the point.
    """
    flat = remove_shadow(_grey(GRADIENT_SHEET, enhanced=False))
    width = flat.shape[1]
    ramp = np.linspace(*SHADOW_RAMP, width, dtype=np.float64)[None, :]
    shadowed = np.clip(flat.astype(np.float64) * ramp, 0, 255).astype(np.uint8)

    left, right = (slice(None), slice(0, width // 3)), (slice(None), slice(2 * width // 3, width))

    def imbalance(binary: np.ndarray) -> str:
        lo = 100 * np.count_nonzero(binary[left]) / binary[left].size
        hi = 100 * np.count_nonzero(binary[right]) / binary[right].size
        return f"shadowed third {lo:.0f}% ink vs lit third {hi:.0f}%"

    otsu_binary, otsu_t = threshold_otsu(shadowed)
    panels = [
        (shadowed, "input: even sheet, linear shadow applied"),
        (threshold_global(shadowed, 127), f"global t=127 — {imbalance(threshold_global(shadowed, 127))}"),
        (otsu_binary, f"otsu t={otsu_t} (optimal, still global) — {imbalance(otsu_binary)}"),
        (threshold_adaptive(shadowed), f"adaptive — {imbalance(threshold_adaptive(shadowed))}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (image, title) in zip(axes.flat, panels):
        _show(ax, image, title)
    fig.suptitle(
        "M4 — why a single global threshold fails on uneven lighting\n"
        "Otsu is the best possible *global* cut and it fails identically; "
        "only a local threshold survives",
        fontsize=11,
    )
    _save(fig, "m4_global_failure.png")


def make_morphology(grey: np.ndarray) -> None:
    """Raw binary, after opening, after closing — whole sheet and zoomed."""
    raw = threshold_adaptive(grey)
    opened = morph_open(raw)
    cleaned = morph_close(opened)
    over_closed = morph_close(opened, kernel_size=7)

    stages = [
        (raw, "raw threshold"),
        (opened, f"after opening (k={config.MORPH_OPEN_K})"),
        (cleaned, f"after closing (k={config.MORPH_CLOSE_K}) — kept"),
        (over_closed, "closing k=7 — strokes fuse to the border"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for column, (image, title) in enumerate(stages):
        components = cv2.connectedComponents(image)[0] - 1
        ink = 100 * np.count_nonzero(image) / image.size
        _show(axes[0, column], image, f"{title}\n{ink:.2f}% ink, {components} components")
        _show(axes[1, column], image[ZOOM], f"{title} (zoom)")

    fig.suptitle(
        "M4 — morphological clean-up. Opening deletes specks, closing repairs "
        "strokes;\nthe fourth column shows the failure that bounds how far "
        "closing can go",
        fontsize=11,
    )
    _save(fig, "m4_morphology.png")


def make_metrics(results_per_sheet: dict[str, list[dict]]) -> None:
    """Grouped bars: ink %, components, line survival and runtime per method."""
    methods = [r["method"] for r in next(iter(results_per_sheet.values()))]
    sheets = sorted(results_per_sheet)

    panels = [
        ("ink_percent", "ink coverage (%)", "lower is cleaner, a few % expected"),
        ("components", "connected components", "fewer = less speckle"),
        ("line_survival", "longest ink run / width", "higher = table lines intact"),
        ("runtime_s", "runtime (s)", "log scale"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    x = np.arange(len(methods))
    bar_width = 0.8 / len(sheets)

    for ax, (key, ylabel, note) in zip(axes.flat, panels):
        for offset, sheet in enumerate(sheets):
            values = [r[key] for r in results_per_sheet[sheet]]
            ax.bar(x + offset * bar_width, values, bar_width, label=sheet.replace(".png", ""))
        ax.set_xticks(x + bar_width * (len(sheets) - 1) / 2)
        ax.set_xticklabels(methods)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — {note}", fontsize=9)
        if key == "runtime_s":
            ax.set_yscale("log")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(sheets), fontsize=8)
    fig.suptitle("M4 — binarisation metrics per method, across all five sheets")
    _save(fig, "m4_metrics.png", rect=(0, 0.05, 1, 1))


def main() -> int:
    config.ensure_dirs()

    sample = _grey(SAMPLE_SHEET)
    make_threshold_comparison(sample)
    make_otsu_histogram(sample)
    make_global_failure()
    make_morphology(sample)

    results_per_sheet = {}
    for path in sorted(config.SHEETS.glob("*.png")):
        results = compare_methods(_grey(path.name))
        results_per_sheet[path.name] = results
        for result in results:
            log.info(
                "%-15s %-9s ink=%5.2f%%  components=%4d  lines=%.3f  %.0f ms",
                path.name,
                result["method"],
                result["ink_percent"],
                result["components"],
                result["line_survival"],
                1000 * result["runtime_s"],
            )
    make_metrics(results_per_sheet)

    print("Wrote 5 figures to outputs/figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
