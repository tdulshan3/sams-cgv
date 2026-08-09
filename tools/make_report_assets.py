#!/usr/bin/env python3
"""Regenerate all figures for the report.

Runs every other make_m*_figures.py script, generates the M9-specific
charts, and draws the chart-choice justification figure.
"""

import subprocess
import sys
import shutil
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.viz.style import apply_house_style, COLOURS
from src.io.db import Database


def run_other_generators():
    """Run all other figure generators."""
    tools_dir = Path(__file__).parent
    
    # M7 requires tune_threshold.py to run first to cache features
    print("Running tune_threshold.py...")
    subprocess.run([sys.executable, str(tools_dir / "tune_threshold.py")])
    for script in sorted(tools_dir.glob("make_m*_figures.py")):
        print(f"Running {script.name}...")
        result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: {script.name} failed with exit code {result.returncode}")
            print(result.stderr)
        else:
            print(f"  {script.name} completed.")


def run_infovis_for_assets():
    """Run infovis.py to generate charts and copy them to figures/."""
    print("Generating M9 charts via infovis.py...")
    # Student dashboard (we use 10000409 as the example)
    subprocess.run([sys.executable, "infovis.py", "10000409", "--save-only"])
    # Class charts
    subprocess.run([sys.executable, "infovis.py", "--all", "--save-only"])

    # Copy to figures/ with M9 prefix
    dash_src = config.CHARTS / "10000409_dashboard.png"
    if dash_src.exists():
        shutil.copy(dash_src, config.FIGURES / "m9_dashboard_example.png")
    
    heat_src = config.CHARTS / "class_heatmap.png"
    if heat_src.exists():
        shutil.copy(heat_src, config.FIGURES / "m9_class_heatmap.png")
        
    dist_src = config.CHARTS / "attendance_distribution.png"
    if dist_src.exists():
        shutil.copy(dist_src, config.FIGURES / "m9_attendance_distribution.png")


def draw_chart_type_choice():
    """Draw a comparison of pie vs bar vs timeline to justify choices."""
    print("Drawing m9_chart_type_choice.png...")
    apply_house_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Fake data: attendance over 5 days
    days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
    present = [1, 1, 0, 1, 1]
    
    # Pie chart (Terrible for this data)
    axes[0].pie(
        [sum(present), len(present) - sum(present)], 
        labels=["Present", "Absent"],
        colors=[COLOURS["present"], COLOURS["absent"]],
        autopct='%1.1f%%'
    )
    axes[0].set_title("Pie Chart\n(Loses temporal order entirely)")
    
    # Bar chart (Okay, but visually heavy)
    axes[1].bar(days, present, color=COLOURS["present"])
    axes[1].set_title("Bar Chart\n(Visually heavy for simple binary data)")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Absent", "Present"])
    
    # Timeline / Step (Best)
    axes[2].step(days, present, where='mid', color='#757575', linewidth=2)
    axes[2].scatter(days, present, color=[COLOURS["present"] if p else COLOURS["absent"] for p in present], s=100, zorder=3)
    axes[2].set_title("Timeline (Chosen)\n(Shows order, clean, easy to read)")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Absent", "Present"])
    
    fig.tight_layout()
    fig.savefig(config.FIGURES / "m9_chart_type_choice.png", dpi=config.CHART_DPI, bbox_inches='tight')
    plt.close(fig)


def draw_accuracy_report():
    """Draw a visual table summarizing accuracy."""
    print("Drawing m9_accuracy_report.png...")
    if not config.GROUND_TRUTH.exists():
        print("  WARNING: Ground truth missing, skipping accuracy report.")
        return

    gt_df = pd.read_csv(config.GROUND_TRUTH, dtype={'student_index': str})
    db = Database()
    records = db.get_all_attendance()
    if not records:
        print("  WARNING: Database empty, skipping accuracy report.")
        return

    db_df = pd.DataFrame(records)
    merged = gt_df.merge(db_df, on=['sheet_date', 'student_index'], suffixes=('_gt', '_db'))
    
    total = len(merged)
    correct = (merged['present_gt'] == merged['present_db']).sum()
    accuracy = correct / total * 100 if total > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    
    # Summary text
    ax.text(0.5, 0.9, "Accuracy Report against Ground Truth", ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 0.7, f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total} cells correct)", ha='center', va='center', fontsize=14)
    
    # Detail table for mistakes
    mistakes = merged[merged['present_gt'] != merged['present_db']]
    
    if mistakes.empty:
        ax.text(0.5, 0.4, "Perfect Match! No misclassifications.", ha='center', va='center', fontsize=12, color=COLOURS["present"])
    else:
        table_data = [["Date", "Student", "Predicted", "Actual", "Note"]]
        for _, row in mistakes.iterrows():
            gt_status = "Present" if row['present_gt'] else "Absent"
            db_status = "Present" if row['present_db'] else "Absent"
            table_data.append([
                row['sheet_date'], 
                row['student_index'], 
                db_status, 
                gt_status,
                str(row['note'])[:30] + "..." if pd.notna(row['note']) and len(str(row['note'])) > 30 else str(row['note'] if pd.notna(row['note']) else "")
            ])
            
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0.05, 0.1, 0.9, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        # Style header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E0E0E0')
                
    fig.savefig(config.FIGURES / "m9_accuracy_report.png", dpi=config.CHART_DPI, bbox_inches='tight')
    plt.close(fig)


def main():
    config.ensure_dirs()
    print("=" * 60)
    print(" Generating all report assets")
    print("=" * 60)
    
    # To get accuracy, we must ensure all sheets have been run recently
    print("Step 1: Running all sheets to populate DB...")
    run_all_cmd = [sys.executable, "tools/run_all_sheets.py"]
    subprocess.run(run_all_cmd)
    
    print("\nStep 2: Generating M9 assets...")
    run_infovis_for_assets()
    draw_chart_type_choice()
    draw_accuracy_report()
    
    print("\nStep 3: Running other asset generators...")
    run_other_generators()
    
    print("\nDone! All figures are in outputs/figures/")

if __name__ == "__main__":
    main()
