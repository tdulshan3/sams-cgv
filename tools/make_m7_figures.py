#!/usr/bin/env python3
"""Generate M7's five report figures.

    python tools/tune_threshold.py        # first, caches the measurements
    python tools/make_m7_figures.py       # then, draws them

Everything here is drawn from ``outputs/decision_features.csv``, the cache one
pipeline pass over all five sheets leaves behind, so the figures in the report
and the threshold in ``config.py`` cannot disagree with each other: they are
the same thirty measurements.

    m7_ink_ratio_distribution.png   the main figure, where present and absent
                                    actually sit, and where the threshold falls
    m7_threshold_sweep.png          accuracy at every threshold, ink alone
                                    against the full rule
    m7_confusion_matrix.png         the verdicts, with and without the two
                                    known ink-but-absent cells
    m7_accuracy_per_sheet.png       which sheet the one error is on
    m7_er_diagram.png               the four tables and their keys
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.colors  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from tools.tune_threshold import (  # noqa: E402
    CellSample,
    accuracy,
    best_plateau,
    confusion,
    load_features,
    predict,
    sweep,
)

log = get_logger("m7_figures")

# Categorical slots 1 and 2 of the house palette, blue and orange, the pair
# that stays separable under every common form of colour blindness. Present and
# absent are identities, not magnitudes, so they take categorical hues; the
# confusion matrix shades counts and so takes a single-hue sequential ramp.
PRESENT = "#2a78d6"
ABSENT = "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def _style() -> None:
    """House chrome: recessive grid and axes, ink-coloured text."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.size": 9,
            "axes.titlesize": 11,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    path = config.FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s", path.relative_to(config.ROOT))


def _rule() -> dict:
    return dict(
        min_stroke=config.MIN_STROKE_LENGTH,
        min_components=config.MIN_COMPONENTS,
        max_ink=config.MAX_INK_RATIO,
    )


def figure_distribution(samples: list[CellSample]) -> None:
    """Where the two groups actually sit on the ink-ratio axis.

    Two panels sharing one axis. The histogram is what the eye reads the shape
    from; the strip below it plots all thirty cells individually, because with
    five absent cells a histogram bar can hide whether a bin holds one cell or
    four, and the whole argument for the threshold is about individual cells.
    """
    present = [s.ink_ratio for s in samples if s.truth == 1]
    absent = [s.ink_ratio for s in samples if s.truth == 0]

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    bins = np.linspace(0, max(s.ink_ratio for s in samples) * 1.05, 34)
    top.hist(present, bins=bins, color=PRESENT, alpha=0.85, label=f"signed ({len(present)})")
    top.hist(absent, bins=bins, color=ABSENT, alpha=0.85, label=f"not signed ({len(absent)})")
    top.axvline(config.INK_RATIO_THRESHOLD, color=INK, linewidth=2)
    top.annotate(
        f"threshold {config.INK_RATIO_THRESHOLD:.4f}",
        xy=(config.INK_RATIO_THRESHOLD, top.get_ylim()[1] * 0.92),
        xytext=(12, 0),
        textcoords="offset points",
        fontsize=9,
        color=INK,
    )
    top.axvline(config.MAX_INK_RATIO, color=INK, linewidth=1.2, linestyle=":")
    top.annotate(
        f"upper bound {config.MAX_INK_RATIO}",
        xy=(config.MAX_INK_RATIO, top.get_ylim()[1] * 0.55),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    top.set_ylabel("cells")
    top.set_title("Ink coverage separates signed from unsigned; except where the mask is wrong")
    top.legend(loc="upper right")

    rng = np.random.default_rng(402)
    for group, colour, y in ((present, PRESENT, 1.0), (absent, ABSENT, 0.6)):
        bottom.scatter(
            group,
            y + rng.uniform(-0.05, 0.05, len(group)),
            s=34,
            color=colour,
            edgecolor=SURFACE,
            linewidth=1.2,
            zorder=3,
        )
    bottom.axvline(config.INK_RATIO_THRESHOLD, color=INK, linewidth=2)

    for sample in samples:
        if sample.is_known_case:
            bottom.annotate(
                f"{sample.sheet_date[:5]} / {sample.student_index}",
                xy=(sample.ink_ratio, 0.6),
                xytext=(0, -26),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=ABSENT,
                arrowprops=dict(arrowstyle="-", color=ABSENT, linewidth=0.9),
            )

    bottom.set_yticks([0.6, 1.0], ["not signed", "signed"], fontsize=8)
    bottom.set_ylim(0.25, 1.25)
    bottom.set_xlabel("ink ratio, ink pixels ÷ cell pixels")
    bottom.grid(axis="y", visible=False)
    _save(fig, "m7_ink_ratio_distribution.png")


def figure_sweep(samples: list[CellSample]) -> None:
    """Accuracy at every candidate threshold, and why the plateau matters."""
    ink_only = dict(min_stroke=0, min_components=0, max_ink=1.0)
    thresholds, ink_scores = sweep(samples, **ink_only)
    _, rule_scores = sweep(samples, **_rule())
    chosen, low, high = best_plateau(thresholds, ink_scores)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(low, high, color=PRESENT, alpha=0.10)
    ax.plot(thresholds, [s * 100 for s in rule_scores], color=PRESENT, linewidth=2,
            label="full rule (ink + components + stroke length)")
    ax.plot(thresholds, [s * 100 for s in ink_scores], color=ABSENT, linewidth=2,
            linestyle="--", label="ink ratio alone")
    ax.axvline(config.INK_RATIO_THRESHOLD, color=INK, linewidth=2)

    ax.annotate(
        f"chosen {config.INK_RATIO_THRESHOLD:.4f}\nmidpoint of the plateau",
        xy=(config.INK_RATIO_THRESHOLD, 55),
        xytext=(28, -6),
        textcoords="offset points",
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=INK, linewidth=1),
    )
    ax.annotate(
        f"plateau {low:.4f} – {high:.4f}\nno labelled cell lies in this gap,\nso every value here scores the same",
        xy=((low + high) / 2, 97),
        xytext=(0.30, 86),
        textcoords="data",
        fontsize=8.5,
        color=MUTED,
        arrowprops=dict(arrowstyle="->", color=MUTED, linewidth=1),
    )
    ax.set_xlabel("INK_RATIO_THRESHOLD")
    ax.set_ylabel("accuracy over 30 labelled cells (%)")
    ax.set_title("The threshold is read off this curve, not chosen by eye")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower left")
    _save(fig, "m7_threshold_sweep.png")


def figure_confusion(samples: list[CellSample]) -> None:
    """The verdicts, counted twice: all 30 cells, then the 28 without the known pair."""
    subsets = [
        ("All 30 cells", samples),
        ("Excluding the 2 ink-but-absent cells", [s for s in samples if not s.is_known_case]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
    fig.subplots_adjust(wspace=0.45)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)

    for ax, (title, subset) in zip(axes, subsets):
        tp, fp, tn, fn = confusion(subset, config.INK_RATIO_THRESHOLD, **_rule())
        matrix = np.array([[tp, fn], [fp, tn]])
        ax.imshow(matrix, cmap=ramp, vmin=0, vmax=matrix.max())
        for (row, column), value in np.ndenumerate(matrix):
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=15,
                color=SURFACE if value > matrix.max() * 0.55 else INK,
            )
        ax.set_xticks([0, 1], ["called present", "called absent"], fontsize=8.5)
        ax.set_yticks([0, 1], ["truly present", "truly absent"], fontsize=8.5)
        ax.set_title(f"{title}\n{(tp + tn)}/{len(subset)} = {(tp + tn) / len(subset):.1%}",
                     fontsize=10)
        ax.grid(visible=False)
        ax.tick_params(length=0)

    fig.suptitle("Every error is a false present; nobody who signed is ever called absent", y=1.10)
    _save(fig, "m7_confusion_matrix.png")


def figure_per_sheet(samples: list[CellSample]) -> None:
    """Which sheet the errors are on, and which cell each one is."""
    dates = sorted(
        {s.sheet_date for s in samples},
        key=lambda d: (d[6:], d[3:5], d[:2]),
    )
    correct, wrong_labels = [], []
    for date in dates:
        rows = [s for s in samples if s.sheet_date == date]
        hits = sum(1 for s in rows if predict(s, config.INK_RATIO_THRESHOLD, **_rule()) == bool(s.truth))
        correct.append(hits / len(rows) * 100)
        misses = [
            s.student_index
            for s in rows
            if predict(s, config.INK_RATIO_THRESHOLD, **_rule()) != bool(s.truth)
        ]
        wrong_labels.append(", ".join(misses))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(dates, correct, color=PRESENT, width=0.6)
    for bar, score, label in zip(bars, correct, wrong_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 1.5,
            f"{score:.0f}%",
            ha="center",
            fontsize=9,
            color=INK,
        )
        if label:
            bar.set_color(ABSENT)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score / 2,
                f"missed\n{label}",
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE,
            )
    ax.set_ylabel("cells correct (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Accuracy per sheet, 6 cells each")
    ax.grid(axis="x", visible=False)
    _save(fig, "m7_accuracy_per_sheet.png")


TABLES = {
    "students": ["student_index  PK", "name"],
    "sheets": ["id  PK", "sheet_date  UNIQUE", "image_path", "subject_code", "processed_at"],
    "attendance": [
        "id  PK",
        "student_index  FK",
        "sheet_id  FK",
        "present",
        "confidence",
        "ink_ratio",
        "UNIQUE(student_index, sheet_id)",
    ],
    "signatures": [
        "id  PK",
        "student_index  FK",
        "sheet_id  FK",
        "crop_path",
        "mask_path",
        "ink_ratio, components,",
        "aspect, stroke_length",
        "UNIQUE(student_index, sheet_id)",
    ],
}

POSITIONS = {
    "students": (0.04, 0.60),
    "sheets": (0.04, 0.12),
    "attendance": (0.55, 0.55),
    "signatures": (0.55, 0.04),
}


def figure_er_diagram() -> None:
    """The four tables, their columns, and which keys point where."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    row_height = 0.052
    boxes: dict[str, tuple[float, float, float, float]] = {}
    for name, columns in TABLES.items():
        x, y = POSITIONS[name]
        width = 0.38
        height = row_height * (len(columns) + 1)
        boxes[name] = (x, y, width, height)
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                facecolor=SURFACE,
                edgecolor=PRESENT,
                linewidth=1.6,
            )
        )
        ax.add_patch(
            patches.Rectangle((x, y + height - row_height), width, row_height, facecolor=PRESENT)
        )
        ax.text(
            x + 0.014,
            y + height - row_height / 2,
            name,
            va="center",
            fontsize=10.5,
            color=SURFACE,
            fontweight="bold",
        )
        for position, column in enumerate(columns):
            ax.text(
                x + 0.014,
                y + height - row_height * (position + 1.5),
                column,
                va="center",
                fontsize=8.2,
                color=INK if "PK" in column or "FK" in column else MUTED,
            )

    def arrow(source: str, target: str, label: str, source_y: float, target_y: float) -> None:
        sx, sy, sw, sh = boxes[source]
        tx, ty, tw, th = boxes[target]
        ax.annotate(
            "",
            xy=(tx + tw, ty + th * target_y),
            xytext=(sx, sy + sh * source_y),
            arrowprops=dict(arrowstyle="-|>", color=ABSENT, linewidth=1.6,
                            connectionstyle="arc3,rad=0.12"),
        )
        ax.text(
            (sx + tx + tw) / 2,
            (sy + sh * source_y + ty + th * target_y) / 2 + 0.02,
            label,
            ha="center",
            fontsize=8,
            color=ABSENT,
        )

    arrow("attendance", "students", "student_index", 0.78, 0.62)
    arrow("attendance", "sheets", "sheet_id", 0.12, 0.92)
    arrow("signatures", "students", "student_index", 0.92, 0.18)
    arrow("signatures", "sheets", "sheet_id", 0.22, 0.30)

    ax.set_title(
        "data/attendance.db; one row per student per sheet, unique on the pair",
        fontsize=11,
        color=INK,
    )
    _save(fig, "m7_er_diagram.png")


def main() -> int:
    _style()
    config.ensure_dirs()
    samples = load_features()
    log.info("%d cells loaded", len(samples))
    figure_distribution(samples)
    figure_sweep(samples)
    figure_confusion(samples)
    figure_per_sheet(samples)
    figure_er_diagram()
    print(
        f"accuracy at INK_RATIO_THRESHOLD={config.INK_RATIO_THRESHOLD}: "
        f"{accuracy(samples, config.INK_RATIO_THRESHOLD, **_rule()):.1%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
