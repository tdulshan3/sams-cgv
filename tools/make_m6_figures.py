"""Generate Module M6's six report figures.

Saves figures to outputs/figures/m6_*.png with minimum 150 DPI.
Usage:
    python tools/make_m6_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.detect.cell_clean import remove_table_lines
from src.detect.ink_mask import (
    InkStage,
    clean_ink_mask,
    count_pen_colours,
    dominant_pen_colour,
    ink_features,
    ink_mask,
    ink_mask_combined,
)
from src.models import SheetMeta, Student
from src.pipeline import Pipeline
from src.preprocess.binarize import BinarizeStage
from src.preprocess.deskew import GeometryStage
from src.preprocess.enhance import EnhanceStage
from src.table.cell_extract import TableStage
from src.utils.logging import get_logger

log = get_logger("make_m6_figures")


def _save_fig(fig: plt.Figure, name: str) -> None:
    path = config.FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved report figure %s", path.relative_to(config.ROOT))


def run_pipeline_on_sheet(sheet_path: Path) -> dict:
    sheet = SheetMeta(path=sheet_path, date=sheet_path.stem)
    students = [Student(index=str(i), name=f"Student {i}") for i in range(1, 7)]
    pipeline = Pipeline([
        GeometryStage(),
        EnhanceStage(),
        BinarizeStage(),
        TableStage(),
        InkStage(),
    ])
    return pipeline.run(sheet, students)


def make_fig1_colour_spaces(sample_cell: np.ndarray) -> None:
    """m6_colour_spaces.png — BGR, H, S, V and LAB a/b channels."""
    hsv = cv2.cvtColor(sample_cell, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(sample_cell, cv2.COLOR_BGR2LAB)

    h_chan = hsv[:, :, 0]
    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]
    a_chan = lab[:, :, 1]
    b_chan = lab[:, :, 2]

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))

    axes[0, 0].imshow(cv2.cvtColor(sample_cell, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("(a) BGR Colour Crop", fontsize=10)

    axes[0, 1].imshow(h_chan, cmap="hsv")
    axes[0, 1].set_title("(b) HSV — Hue (H)", fontsize=10)

    axes[0, 2].imshow(s_chan, cmap="magma")
    axes[0, 2].set_title("(c) HSV — Saturation (S)", fontsize=10)

    axes[1, 0].imshow(v_chan, cmap="gray")
    axes[1, 0].set_title("(d) HSV — Value (V)", fontsize=10)

    axes[1, 1].imshow(a_chan, cmap="coolwarm")
    axes[1, 1].set_title("(e) LAB — a* Channel", fontsize=10)

    axes[1, 2].imshow(b_chan, cmap="coolwarm")
    axes[1, 2].set_title("(f) LAB — b* Channel", fontsize=10)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("M6 Figure 1: Colour Space Decomposition of Signature Cell Crop", fontsize=12, fontweight="bold")
    _save_fig(fig, "m6_colour_spaces.png")


def make_fig2_hue_scatter(sheets_data: list[dict]) -> None:
    """m6_hue_scatter.png — Scatter of hue vs saturation for ink pixels."""
    hues_by_color = {"blue": [], "black": [], "red": [], "green": [], "other": []}
    sats_by_color = {"blue": [], "black": [], "red": [], "green": [], "other": []}

    for ctx in sheets_data:
        ink_results = ctx.get("ink", [])
        for res in ink_results:
            if res.cell.image is not None and res.mask is not None and np.any(res.mask):
                hsv = cv2.cvtColor(res.cell.image, cv2.COLOR_BGR2HSV)
                ink_pts = hsv[res.mask > 0]
                if len(ink_pts) > 0:
                    color = dominant_pen_colour(res.cell.image, res.mask)
                    # Sample up to 200 pixels per cell for clean scatter
                    idx = np.random.choice(len(ink_pts), size=min(200, len(ink_pts)), replace=False)
                    hues_by_color[color].extend(ink_pts[idx, 0])
                    sats_by_color[color].extend(ink_pts[idx, 1])

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {"blue": "#1f77b4", "black": "#2ca02c", "red": "#d62728", "green": "#ff7f0e", "other": "#7f7f7f"}
    labels = {"blue": "Blue Pen", "black": "Black Pen", "red": "Red Pen", "green": "Green Pen", "other": "Other/Unclassified"}

    for color in ["blue", "black", "red", "green", "other"]:
        if hues_by_color[color]:
            ax.scatter(
                hues_by_color[color],
                sats_by_color[color],
                c=palette[color],
                label=labels[color],
                alpha=0.6,
                s=15,
                edgecolors="none",
            )

    ax.axhline(config.SAT_MIN, color="gray", linestyle="--", linewidth=1, label=f"SAT_MIN threshold ({config.SAT_MIN})")
    ax.set_xlabel("Hue (H)", fontsize=11)
    ax.set_ylabel("Saturation (S)", fontsize=11)
    ax.set_title("M6 Figure 2: Ink Pixel Hue vs Saturation Scatter Plot across Signing Sheets", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    _save_fig(fig, "m6_hue_scatter.png")


def make_fig3_mask_panels(sheets_data: list[dict]) -> None:
    """m6_mask_panels.png — 6 cells: crop -> HSV mask -> cleaned mask."""
    all_res = []
    for ctx in sheets_data:
        all_res.extend(ctx.get("ink", []))

    cells_to_show = all_res[:6]
    fig, axes = plt.subplots(len(cells_to_show), 3, figsize=(9, 2 * len(cells_to_show)))

    for i, res in enumerate(cells_to_show):
        cell_crop = res.cell.image if res.cell.image is not None else np.zeros((40, 80, 3), dtype=np.uint8)
        cleaned_bgr = remove_table_lines(cell_crop)
        raw_mask = ink_mask_combined(cleaned_bgr)
        final_mask = clean_ink_mask(raw_mask)

        axes[i, 0].imshow(cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_ylabel(f"Row {res.cell.row + 1}", fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        axes[i, 1].imshow(raw_mask, cmap="gray")
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        axes[i, 2].imshow(final_mask, cmap="gray")
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        if i == 0:
            axes[i, 0].set_title("(a) Cleaned Crop", fontsize=10)
            axes[i, 1].set_title("(b) Raw HSV Mask", fontsize=10)
            axes[i, 2].set_title("(c) Cleaned Mask", fontsize=10)

    fig.suptitle("M6 Figure 3: Cell Crop to Segmentation Mask Pipeline Stages", fontsize=12, fontweight="bold")
    _save_fig(fig, "m6_mask_panels.png")


def make_fig4_border_removal(sample_cell: np.ndarray) -> None:
    """m6_border_removal.png — Before and after removing leftover table lines."""
    raw_crop = sample_cell.copy()

    # Add artificial dark border line to demonstrate line removal clearly
    raw_crop[:4, :] = 30
    raw_crop[:, :4] = 30

    cleaned_crop = remove_table_lines(raw_crop)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title("(a) Before: Cell Crop with Table Border Lines", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(cleaned_crop, cv2.COLOR_BGR2RGB))
    axes[1].set_title("(b) After: Table Border Lines Erased", fontsize=10)
    axes[1].axis("off")

    fig.suptitle("M6 Figure 4: Leftover Table Line Removal Demonstration", fontsize=12, fontweight="bold")
    _save_fig(fig, "m6_border_removal.png")


def make_fig5_pen_colour_counts(sheets_data: list[dict]) -> None:
    """m6_pen_colour_counts.png — Bar chart of pen colours used per sheet."""
    sheet_dates = [ctx["sheet"].date for ctx in sheets_data]
    colors = ["blue", "black", "red", "green"]

    counts_per_sheet = {c: [] for c in colors}

    for ctx in sheets_data:
        cells = [r.cell.image for r in ctx.get("ink", []) if r.cell.image is not None]
        masks = [r.mask for r in ctx.get("ink", []) if r.mask is not None]
        tally = count_pen_colours(cells, masks)
        for c in colors:
            counts_per_sheet[c].append(tally.get(c, 0))

    x = np.arange(len(sheet_dates))
    width = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = {"blue": "#1f77b4", "black": "#333333", "red": "#d62728", "green": "#2ca02c"}

    for i, c in enumerate(colors):
        ax.bar(x + i * width, counts_per_sheet[c], width, label=c.capitalize(), color=bar_colors[c])

    ax.set_xlabel("Signing Sheet Date", fontsize=11)
    ax.set_ylabel("Count of Signed Cells", fontsize=11)
    ax.set_title("M6 Figure 5: Dominant Pen Colour Usage Per Signing Sheet", fontsize=12, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(sheet_dates)
    ax.legend(title="Pen Colour")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

    _save_fig(fig, "m6_pen_colour_counts.png")


def make_fig6_empty_vs_signed(sheets_data: list[dict]) -> None:
    """m6_empty_vs_signed.png — An empty cell and a signed cell with ink ratios."""
    all_res = []
    for ctx in sheets_data:
        all_res.extend(ctx.get("ink", []))

    # Find an empty cell (lowest ink ratio) and a signed cell (highest ink ratio)
    all_res_sorted = sorted(all_res, key=lambda r: r.ink_ratio)
    empty_res = all_res_sorted[0]
    signed_res = all_res_sorted[-1]

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))

    empty_img = empty_res.cell.image if empty_res.cell.image is not None else np.full((40, 80, 3), 245, dtype=np.uint8)
    signed_img = signed_res.cell.image if signed_res.cell.image is not None else np.full((40, 80, 3), 245, dtype=np.uint8)

    empty_mask = empty_res.mask if empty_res.mask is not None else np.zeros(empty_img.shape[:2], dtype=np.uint8)
    signed_mask = signed_res.mask if signed_res.mask is not None else np.zeros(signed_img.shape[:2], dtype=np.uint8)

    axes[0, 0].imshow(cv2.cvtColor(empty_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f"(a) Empty Cell Crop\nink_ratio = {empty_res.ink_ratio:.4f}", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(empty_mask, cmap="gray")
    axes[0, 1].set_title(f"(b) Empty Cell Ink Mask\ncomponents = {empty_res.components}", fontsize=10)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(cv2.cvtColor(signed_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"(c) Signed Cell Crop\nink_ratio = {signed_res.ink_ratio:.4f}", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(signed_mask, cmap="gray")
    axes[1, 1].set_title(f"(d) Signed Cell Ink Mask\ncomponents = {signed_res.components}", fontsize=10)
    axes[1, 1].axis("off")

    fig.suptitle("M6 Figure 6: Comparison of Empty vs Signed Signature Cells", fontsize=12, fontweight="bold")
    _save_fig(fig, "m6_empty_vs_signed.png")


def main() -> int:
    config.ensure_dirs()
    sheets = sorted(config.SHEETS.glob("*.png"))
    if not sheets:
        log.error("No sheets found in %s", config.SHEETS)
        return 1

    log.info("Processing %d sheets to generate M6 report figures...", len(sheets))
    sheets_data = []
    for sheet_path in sheets:
        log.info("Running pipeline on sheet %s", sheet_path.name)
        ctx = run_pipeline_on_sheet(sheet_path)
        sheets_data.append(ctx)

    sample_cell = cv2.imread(str(config.FIXTURES / "cell_sample.png"))
    if sample_cell is None and sheets_data:
        # Fallback to first cell image from first sheet
        sample_cell = sheets_data[0]["ink"][0].cell.image

    log.info("Generating figure 1: m6_colour_spaces.png...")
    make_fig1_colour_spaces(sample_cell)

    log.info("Generating figure 2: m6_hue_scatter.png...")
    make_fig2_hue_scatter(sheets_data)

    log.info("Generating figure 3: m6_mask_panels.png...")
    make_fig3_mask_panels(sheets_data)

    log.info("Generating figure 4: m6_border_removal.png...")
    make_fig4_border_removal(sample_cell)

    log.info("Generating figure 5: m6_pen_colour_counts.png...")
    make_fig5_pen_colour_counts(sheets_data)

    log.info("Generating figure 6: m6_empty_vs_signed.png...")
    make_fig6_empty_vs_signed(sheets_data)

    log.info("All 6 M6 report figures generated successfully in %s", config.FIGURES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
