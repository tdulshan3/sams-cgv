"""Draw M5's seven report figures (BUILD_SPEC.md section 9.5).

    python tools/make_m5_figures.py [sheet.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.models import SheetMeta  # noqa: E402
from src.preprocess.binarize import BinarizeStage  # noqa: E402
from src.preprocess.deskew import GeometryStage  # noqa: E402
from src.preprocess.enhance import EnhanceStage  # noqa: E402
from src.table.cell_extract import TableStage, _clip_to_table_width  # noqa: E402
from src.table.grid_builder import (  # noqa: E402
    _group_into_bands,
    longest_regular_run,
    select_student_table,
)
from src.table.line_detect import (  # noqa: E402
    detect_horizontal_lines,
    detect_lines_hough,
    detect_vertical_lines,
    line_mask,
)
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("m5figures")


def _rgb(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _panel(axis, image: np.ndarray, title: str) -> None:
    axis.imshow(_rgb(image), cmap="gray" if image.ndim == 2 else None)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _save(figure, name: str) -> None:
    path = config.FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote %s", path.relative_to(config.ROOT))


def build_context(sheet: Path) -> tuple[dict, TableStage]:
    """Run the real pipeline up to and including table detection."""
    ctx: dict = {"sheet": SheetMeta(path=sheet, date=sheet.stem), "students": []}
    for stage in (GeometryStage(), EnhanceStage(), BinarizeStage()):
        ctx = stage.run(ctx)
    table = TableStage()
    ctx = table.run(ctx)
    return ctx, table


def line_masks(ctx: dict) -> None:
    binary = ctx["binary"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=config.FIGURE_DPI)
    _panel(axes[0], binary, "binary (ink = 255)")
    _panel(axes[1], line_mask(binary, "horizontal"), "horizontal mask\nwide kernel keeps row rules")
    _panel(axes[2], line_mask(binary, "vertical"), "vertical mask\ntall kernel keeps column rules")
    figure.suptitle("M5, isolating table lines by morphology", fontsize=12)
    figure.tight_layout()
    _save(figure, "m5_line_masks.png")


def projection_profiles(ctx: dict) -> None:
    binary = ctx["binary"]
    grid = ctx["grid"]
    h_profile = line_mask(binary, "horizontal").sum(axis=1) / 255.0

    top, bottom = grid.ys[0], grid.ys[-1]
    strip = binary[top : bottom + 1, :]
    v_profile = line_mask(strip, "vertical").sum(axis=0) / 255.0

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=config.FIGURE_DPI)
    axes[0].plot(h_profile, linewidth=0.9)
    axes[0].plot(grid.ys, h_profile[grid.ys], "rv", markersize=7, label="detected rows")
    axes[0].set_title("horizontal projection profile; peaks are row rules", fontsize=10)
    axes[0].set_xlabel("y (pixels)")
    axes[0].set_ylabel("ink pixels in row")
    axes[0].legend(fontsize=8)

    axes[1].plot(v_profile, linewidth=0.9, color="tab:green")
    axes[1].plot(grid.xs, v_profile[grid.xs], "rv", markersize=7, label="detected columns")
    axes[1].set_title("vertical projection profile, inside the student table", fontsize=10)
    axes[1].set_xlabel("x (pixels)")
    axes[1].set_ylabel("ink pixels in column")
    axes[1].legend(fontsize=8)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("M5, line positions from projection profiles", fontsize=12)
    figure.tight_layout()
    _save(figure, "m5_projection_profiles.png")


def grid_overlay(ctx: dict, table: TableStage) -> None:
    figure, axis = plt.subplots(figsize=(7, 9), dpi=config.FIGURE_DPI)
    _panel(axis, table._grid_overlay, "green = columns, red = rows")
    figure.suptitle("M5, detected grid over the corrected sheet", fontsize=12)
    figure.tight_layout()
    _save(figure, "m5_grid_overlay.png")


def two_tables(ctx: dict) -> None:
    """The header table and the student table, told apart."""
    warped = ctx["warped"].copy()
    all_ys = detect_horizontal_lines(ctx["binary"])
    student = set(select_student_table(_group_into_bands(all_ys)))

    for y in all_ys:
        chosen = y in student
        colour = (0, 200, 0) if chosen else (0, 140, 255)
        cv2.line(warped, (0, y), (warped.shape[1], y), colour, 3)

    figure, axis = plt.subplots(figsize=(7, 9), dpi=config.FIGURE_DPI)
    _panel(
        axis,
        warped,
        "green = student table (evenly spaced run)\n"
        "orange = discarded, including the lecture header table\n"
        "taking the wrong one reports the lecturer's signature as a student's",
    )
    figure.suptitle("M5; there are two tables on the page", fontsize=12)
    figure.tight_layout()
    _save(figure, "m5_two_tables.png")


def cells_numbered(ctx: dict) -> None:
    warped = ctx["warped"].copy()
    for cell in ctx["cells"]:
        x, y, w, h = cell.bbox
        cv2.rectangle(warped, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(
            warped, f"row {cell.row}", (x + 6, y + h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
        )
    figure, axis = plt.subplots(figsize=(7, 9), dpi=config.FIGURE_DPI)
    _panel(axis, warped, f"{len(ctx['cells'])} signature cells, column {config.SIGNATURE_COL}")
    figure.suptitle("M5; the extracted signature cells", fontsize=12)
    figure.tight_layout()
    _save(figure, "m5_cells_numbered.png")


def grid_repair(ctx: dict) -> None:
    """What the regular-run selector rescues from the raw line list."""
    all_ys = detect_horizontal_lines(ctx["binary"])
    kept = longest_regular_run(all_ys)
    dropped = [y for y in all_ys if y not in kept]

    figure, axis = plt.subplots(figsize=(11, 3.4), dpi=config.FIGURE_DPI)
    axis.eventplot([all_ys], colors="lightgrey", lineoffsets=1, linelengths=0.7, linewidths=2)
    axis.eventplot([dropped], colors="tab:orange", lineoffsets=0, linelengths=0.7, linewidths=2)
    axis.eventplot([kept], colors="tab:green", lineoffsets=-1, linelengths=0.7, linewidths=2)
    axis.set_yticks([1, 0, -1])
    axis.set_yticklabels(
        [f"all lines ({len(all_ys)})", f"discarded ({len(dropped)})", f"student table ({len(kept)})"],
        fontsize=9,
    )
    axis.set_xlabel("y (pixels)")
    axis.spines[["top", "right", "left"]].set_visible(False)
    gaps = np.diff(kept) if len(kept) > 1 else np.array([0])
    figure.suptitle(
        f"M5, selecting the table by spacing regularity (kept gaps: {gaps.min()}-{gaps.max()} px)",
        fontsize=11,
    )
    figure.tight_layout()
    _save(figure, "m5_grid_repair.png")


def hough_vs_morphology(ctx: dict) -> None:
    binary = ctx["binary"]
    grid = ctx["grid"]
    hough_ys, hough_xs = detect_lines_hough(binary)

    morph = ctx["warped"].copy()
    for y in grid.ys:
        cv2.line(morph, (0, y), (morph.shape[1], y), (0, 0, 255), 3)
    for x in grid.xs:
        cv2.line(morph, (x, 0), (x, morph.shape[0]), (0, 200, 0), 3)

    hough = ctx["warped"].copy()
    for y in hough_ys:
        cv2.line(hough, (0, y), (hough.shape[1], y), (0, 0, 255), 3)
    for x in hough_xs:
        cv2.line(hough, (x, 0), (x, hough.shape[0]), (0, 200, 0), 3)

    figure, axes = plt.subplots(1, 2, figsize=(10, 7), dpi=config.FIGURE_DPI)
    _panel(axes[0], morph, f"morphology (primary)\n{len(grid.ys)} rows, {len(grid.xs)} columns")
    _panel(axes[1], hough, f"HoughLinesP (cross-check)\n{len(hough_ys)} rows, {len(hough_xs)} columns")
    figure.suptitle("M5, why morphology beats Hough on printed tables", fontsize=12)
    figure.tight_layout()
    _save(figure, "m5_hough_vs_morphology.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_m5_figures.py")
    parser.add_argument("sheet", nargs="?", help="sheet filename, e.g. 31.05.2019.png")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    sheets = sorted(config.SHEETS.glob("*.png"))
    if not sheets:
        print(f"error: no sheets in {config.SHEETS}")
        return 2
    chosen = config.SHEETS / args.sheet if args.sheet else sheets[0]
    if not chosen.is_file():
        print(f"error: no sheet called {args.sheet}")
        return 2

    ctx, table = build_context(chosen)
    line_masks(ctx)
    projection_profiles(ctx)
    grid_overlay(ctx, table)
    two_tables(ctx)
    cells_numbered(ctx)
    grid_repair(ctx)
    hough_vs_morphology(ctx)

    print(f"\nM5 figures for {chosen.stem} written to {config.FIGURES.relative_to(config.ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
