"""Draw the three figures M1 owes the report.

``outputs/figures/m1_architecture.png``
    Block diagram of the pipeline: what each stage does, who owns it, and what
    key it writes into the shared context.

``outputs/figures/m1_montage_<date>.png``
    The full step-by-step montage for one sheet; the picture the brief asks
    for when it says to show the progress of the image processing.

``outputs/figures/m1_timing.png``
    Seconds spent in each stage, so the discussion section can talk about cost
    rather than guess at it.

Usage:

    python tools/make_m1_figures.py                     # first sheet
    python tools/make_m1_figures.py 12.07.2019.png      # a named sheet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.models import SheetMeta  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.viz.progress import ProgressViewer  # noqa: E402

log = get_logger("m1figures")

STAGE_BLOCKS = [
    ("geometry", "M2", "flatten the photo", "warped"),
    ("enhance", "M3", "kill shadow, boost contrast", "grey"),
    ("binarize", "M4", "ink 255, paper 0", "binary"),
    ("table", "M5", "find lines, cut out cells", "cells"),
    ("ink", "M6", "measure ink in each cell", "ink"),
    ("decision", "M7", "present or absent, then store", "records"),
]
"""``(stage, owner, what it does, context key it writes)`` in pipeline order."""

BLOCK_FILL = "#e8eef7"
BLOCK_EDGE = "#2b4c7e"
SIDE_FILL = "#f3efe4"
SIDE_EDGE = "#8a7a52"


def draw_architecture(path: Path) -> Path:
    """Block diagram of the pipeline, one box per stage."""
    rows = len(STAGE_BLOCKS)
    figure, axis = plt.subplots(figsize=(9.5, 1.15 * rows + 2.2), dpi=config.FIGURE_DPI)
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 1.15 * rows + 2.0)
    axis.axis("off")

    top = 1.15 * rows + 1.4
    axis.text(
        5.0, top + 0.35, "SAMS pipeline", ha="center", va="center", fontsize=15
    )
    axis.text(
        5.0,
        top - 0.05,
        "one photo in, one set of attendance records out",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )

    _rounded(axis, 2.6, top - 0.95, 4.8, 0.6, SIDE_FILL, SIDE_EDGE)
    axis.text(
        5.0, top - 0.65, "signing sheet photo  +  info.xml",
        ha="center", va="center", fontsize=10,
    )

    for position, (stage, owner, what, key) in enumerate(STAGE_BLOCKS):
        y = top - 1.9 - position * 1.15
        _arrow(axis, 5.0, y + 0.62, 0.32)
        _rounded(axis, 1.7, y - 0.28, 6.6, 0.78, BLOCK_FILL, BLOCK_EDGE)
        axis.text(2.0, y + 0.22, f"{position + 1}.", fontsize=10, color="#2b4c7e")
        axis.text(2.5, y + 0.22, stage, fontsize=11, va="center")
        axis.text(2.5, y - 0.02, what, fontsize=8.5, va="center", color="#444444")
        axis.text(8.15, y + 0.22, owner, fontsize=9, ha="right", color="#2b4c7e")
        axis.text(
            8.15, y - 0.02, f'ctx["{key}"]', fontsize=8, ha="right",
            family="monospace", color="#666666",
        )

    bottom = top - 1.9 - rows * 1.15
    _arrow(axis, 5.0, bottom + 0.62, 0.32)
    _rounded(axis, 2.6, bottom - 0.28, 4.8, 0.6, SIDE_FILL, SIDE_EDGE)
    axis.text(
        5.0, bottom + 0.02, "attendance.db   →   infovis.py, investigate.py",
        ha="center", va="center", fontsize=10,
    )

    axis.text(
        5.0, bottom - 0.75,
        "every stage subclasses Stage; the runner owns the order and nothing else",
        ha="center", va="center", fontsize=8.5, color="#777777",
    )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote %s", path.relative_to(config.ROOT))
    return path


def _rounded(axis, x, y, width, height, fill, edge) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=fill, edgecolor=edge, linewidth=1.2,
        )
    )


def _arrow(axis, x, y, length) -> None:
    """A downward arrow between two blocks.

    Drawn with ``annotate`` rather than a patch because the axes are not equal
    aspect: a patch arrow's width is in x units and its length in y units, so
    it comes out as a hairline.
    """
    axis.annotate(
        "",
        xy=(x, y - length),
        xytext=(x, y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": BLOCK_EDGE,
            "linewidth": 1.4,
            "mutation_scale": 16,
        },
    )


def draw_timing(timings: dict[str, float], path: Path) -> Path:
    """Horizontal bar chart of seconds per stage."""
    names = list(timings)
    seconds = [timings[name] for name in names]

    figure, axis = plt.subplots(figsize=(7.5, 0.55 * len(names) + 1.8), dpi=config.FIGURE_DPI)
    bars = axis.barh(names, seconds, color=BLOCK_FILL, edgecolor=BLOCK_EDGE)
    axis.invert_yaxis()
    axis.set_xlabel("seconds")
    axis.set_title(f"Time per stage, total {sum(seconds):.2f} s")
    axis.spines[["top", "right"]].set_visible(False)

    widest = max(seconds) if seconds else 1.0
    for bar, value in zip(bars, seconds):
        axis.text(
            bar.get_width() + widest * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} s",
            va="center",
            fontsize=9,
        )
    axis.set_xlim(0, widest * 1.25 if widest else 1.0)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote %s", path.relative_to(config.ROOT))
    return path


def _pick_sheet(name: str | None) -> Path:
    sheets = sorted(config.SHEETS.glob("*.png"))
    if not sheets:
        print(f"error: no sheets in {config.SHEETS}")
        raise SystemExit(2)
    if name is None:
        return sheets[0]
    chosen = config.SHEETS / name
    if not chosen.is_file():
        print(f"error: no sheet called {name}")
        print("       available: " + ", ".join(path.name for path in sheets))
        raise SystemExit(2)
    return chosen


def main(argv: list[str] | None = None) -> int:
    """Draw all three figures. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="make_m1_figures.py",
        description="Draw the architecture, montage and timing figures for the report.",
    )
    parser.add_argument("sheet", nargs="?", help="sheet filename, e.g. 12.07.2019.png")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    sheet_path = _pick_sheet(args.sheet)
    sheet = SheetMeta(path=sheet_path, date=sheet_path.stem)

    draw_architecture(config.FIGURES / "m1_architecture.png")

    # Imported here so the architecture diagram can still be drawn on a machine
    # where the stage list is mid-swap during integration.
    from sams import STAGES, load_students

    viewer = ProgressViewer(sheet.date, show=False, save=True)
    pipeline = Pipeline([make_stage() for make_stage in STAGES], viewer=viewer)
    students, _subject = load_students(config.INFO_XML)
    pipeline.run(sheet, students)
    viewer.save_all()
    viewer.save_montage(config.FIGURES / f"m1_montage_{sheet.date}.png")

    draw_timing(pipeline.timings(), config.FIGURES / "m1_timing.png")

    print()
    print(f"figures for sheet {sheet.date} written to outputs/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
