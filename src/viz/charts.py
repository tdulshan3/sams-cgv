"""M9 Visualisation & QA — Attendance charts.

Provides the visualisations for individual student records and class-wide
summaries, drawn using Matplotlib and Seaborn.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from src.io.db import Database
from src.viz.style import apply_house_style, COLOURS
from src import config
from src.utils.logging import get_logger

log = get_logger("charts")


class AttendanceCharts:
    """Generates attendance visualisations from database records."""

    def __init__(self, db: Database) -> None:
        self.db = db
        apply_house_style()

    def student_timeline(self, index: str) -> Figure:
        """A timeline showing present/absent over time for one student."""
        records = self.db.get_attendance(index)
        student = self.db.get_student(index)
        name = student["name"] if student else "Unknown"
        
        if not records:
            fig, ax = plt.subplots(figsize=config.CHART_FIGSIZE)
            ax.text(0.5, 0.5, "No attendance records found.", ha='center', va='center')
            ax.set_title(f"Timeline: {name} ({index})")
            return fig

        dates = [r["sheet_date"] for r in records]
        # Map present to 1, absent to 0
        status = [1 if r["present"] else 0 for r in records]
        
        # Colors for points
        colors = []
        for r in records:
            if r["confidence"] < config.UNCERTAIN_BELOW:
                colors.append(COLOURS["uncertain"])
            elif r["present"]:
                colors.append(COLOURS["present"])
            else:
                colors.append(COLOURS["absent"])

        fig, ax = plt.subplots(figsize=config.CHART_FIGSIZE)
        
        # Step plot for the timeline
        ax.step(dates, status, where='mid', color='#757575', alpha=0.5, linewidth=2)
        
        # Scatter for individual data points
        ax.scatter(dates, status, color=colors, s=150, zorder=3, edgecolor='white', linewidth=2)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Absent', 'Present'])
        ax.set_ylim(-0.2, 1.2)
        
        ax.set_title(f"Attendance Timeline: {name} ({index})")
        ax.set_xlabel("Date")
        
        fig.tight_layout()
        return fig

    def student_summary(self, index: str) -> Figure:
        """A donut chart of attendance % with class average for context."""
        records = self.db.get_attendance(index)
        all_records = self.db.get_all_attendance()
        student = self.db.get_student(index)
        name = student["name"] if student else "Unknown"

        fig, ax = plt.subplots(figsize=(6, 6))

        if not records or not all_records:
            ax.text(0.5, 0.5, "No data available.", ha='center', va='center')
            ax.set_title(f"Summary: {name}")
            return fig

        present = sum(1 for r in records if r["present"])
        total = len(records)
        
        class_present = sum(1 for r in all_records if r["present"])
        class_total = len(all_records)
        class_avg = (class_present / class_total) * 100 if class_total > 0 else 0

        # Donut chart
        sizes = [present, total - present]
        colors = [COLOURS["present"], COLOURS["absent"]]
        
        wedges, _ = ax.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.3, edgecolor='white')
        )
        
        # Center text
        ax.text(0, 0.1, f"{present} / {total}", ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0, -0.15, f"Class Avg: {class_avg:.1f}%", ha='center', va='center', fontsize=12, color='#616161')
        
        ax.set_title(f"Attendance Summary: {name}")
        fig.tight_layout()
        return fig

    def student_dashboard(self, index: str) -> Figure:
        """Four-panel dashboard for a single student."""
        records = self.db.get_attendance(index)
        all_records = self.db.get_all_attendance()
        student = self.db.get_student(index)
        name = student["name"] if student else "Unknown"

        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f"Attendance Dashboard: {name} ({index})", fontsize=18, fontweight='bold', y=0.95)

        if not records:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No attendance records found.", ha='center', va='center', fontsize=14)
            return fig

        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1])

        # 1. Timeline (Top Left)
        ax_timeline = fig.add_subplot(gs[0, 0])
        dates = [r["sheet_date"] for r in records]
        status = [1 if r["present"] else 0 for r in records]
        colors = [COLOURS["uncertain"] if r["confidence"] < config.UNCERTAIN_BELOW else (COLOURS["present"] if r["present"] else COLOURS["absent"]) for r in records]
        
        ax_timeline.step(dates, status, where='mid', color='#757575', alpha=0.5, linewidth=2)
        ax_timeline.scatter(dates, status, color=colors, s=120, zorder=3, edgecolor='white', linewidth=1.5)
        ax_timeline.set_yticks([0, 1])
        ax_timeline.set_yticklabels(['Absent', 'Present'])
        ax_timeline.set_ylim(-0.2, 1.2)
        ax_timeline.set_title("Timeline", fontweight='bold')

        # 2. Donut (Top Right)
        ax_donut = fig.add_subplot(gs[0, 1])
        present = sum(1 for r in records if r["present"])
        total = len(records)
        class_present = sum(1 for r in all_records if r["present"])
        class_total = len(all_records)
        class_avg = (class_present / class_total) * 100 if class_total > 0 else 0

        wedges, _ = ax_donut.pie(
            [present, total - present],
            colors=[COLOURS["present"], COLOURS["absent"]],
            startangle=90,
            wedgeprops=dict(width=0.4, edgecolor='white')
        )
        ax_donut.text(0, 0.1, f"{present} / {total}", ha='center', va='center', fontsize=20, fontweight='bold')
        ax_donut.text(0, -0.2, f"Class Avg: {class_avg:.1f}%", ha='center', va='center', fontsize=11, color='#616161')
        ax_donut.set_title("Summary", fontweight='bold')

        # 3. vs Class (Bottom Left)
        ax_vs = fig.add_subplot(gs[1, 0])
        student_pct = (present / total) * 100 if total > 0 else 0
        
        bars = ax_vs.bar(['Student', 'Class Average'], [student_pct, class_avg], color=[COLOURS["present"], '#9E9E9E'], width=0.5)
        ax_vs.set_ylim(0, 100)
        ax_vs.set_ylabel("Attendance %")
        ax_vs.set_title("Comparison", fontweight='bold')
        for bar in bars:
            yval = bar.get_height()
            ax_vs.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=10)

        # 4. Table (Bottom Right)
        ax_table = fig.add_subplot(gs[1, 1])
        ax_table.axis('off')
        
        table_data = []
        for r in records:
            stat_str = "Present" if r["present"] else "Absent"
            if r["confidence"] < config.UNCERTAIN_BELOW:
                stat_str += " (?)"
            table_data.append([r["sheet_date"], stat_str, f"{r['confidence']:.2f}", f"{r['ink_ratio']:.3f}"])
            
        table = ax_table.table(
            cellText=table_data,
            colLabels=["Date", "Status", "Confidence", "Ink Ratio"],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        ax_table.set_title("Details", fontweight='bold')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def class_heatmap(self) -> Figure:
        """A heatmap of all students vs dates."""
        records = self.db.get_all_attendance()
        fig, ax = plt.subplots(figsize=(10, 6))

        if not records:
            ax.text(0.5, 0.5, "No attendance records found.", ha='center', va='center')
            ax.set_title("Class Attendance Heatmap")
            return fig

        df = pd.DataFrame(records)
        df['label'] = df['name'] + " (" + df['student_index'] + ")"
        
        # Create a pivot table: rows=students, cols=dates, values=present
        pivot = df.pivot(index='label', columns='sheet_date', values='present')
        
        # Values for custom colors: 1 for present, 0 for absent. We will also check confidence in a real robust system,
        # but for the visual grid a simple 1/0 is fine.
        cmap = sns.color_palette([COLOURS["absent"], COLOURS["present"]], as_cmap=True)
        
        sns.heatmap(pivot, cmap=cmap, cbar=False, annot=True, fmt="g", linewidths=1, ax=ax, 
                    annot_kws={'color': 'white', 'weight': 'bold'})
        
        # To display "P" / "A" instead of 1/0
        for t in ax.texts:
            if t.get_text() == '1':
                t.set_text('P')
            elif t.get_text() == '0':
                t.set_text('A')

        ax.set_title("Class Attendance Heatmap", pad=20)
        ax.set_xlabel("Date")
        ax.set_ylabel("Student")
        ax.tick_params(axis='x', rotation=0)
        
        fig.tight_layout()
        return fig

    def sheet_totals(self) -> Figure:
        """Bar chart showing number of present students per sheet."""
        records = self.db.get_all_attendance()
        fig, ax = plt.subplots(figsize=(8, 5))

        if not records:
            ax.text(0.5, 0.5, "No attendance records found.", ha='center', va='center')
            return fig

        df = pd.DataFrame(records)
        totals = df.groupby('sheet_date')['present'].sum()
        
        bars = ax.bar(totals.index, totals.values, color=COLOURS["present"], width=0.6)
        
        ax.set_ylim(0, df['student_index'].nunique() + 1)
        ax.set_title("Present Count per Sheet")
        ax.set_xlabel("Date")
        ax.set_ylabel("Students Present")
        
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')

        fig.tight_layout()
        return fig

    def attendance_distribution(self) -> Figure:
        """Histogram of attendance percentage across the class."""
        records = self.db.get_all_attendance()
        fig, ax = plt.subplots(figsize=(8, 5))

        if not records:
            ax.text(0.5, 0.5, "No attendance records found.", ha='center', va='center')
            return fig

        df = pd.DataFrame(records)
        # Calculate % present for each student
        student_pct = df.groupby('student_index')['present'].mean() * 100

        sns.histplot(student_pct, bins=np.arange(0, 110, 10), color='#1976D2', ax=ax)
        
        ax.set_title("Attendance Distribution")
        ax.set_xlabel("Attendance %")
        ax.set_ylabel("Number of Students")
        ax.set_xlim(-5, 105)
        
        # Force integer y-ticks
        ax.yaxis.get_major_locator().set_params(integer=True)
        
        fig.tight_layout()
        return fig

    def save(self, fig: Figure, name: str) -> Path:
        """Saves a figure to the charts output directory."""
        path = config.CHARTS / f"{name}.png"
        fig.savefig(path, dpi=config.CHART_DPI, bbox_inches='tight')
        log.info("saved chart %s", path)
        return path


def show_student(index: str, save_only: bool = False) -> None:
    """Entry point for `infovis.py <index>`."""
    db = Database()
    charts = AttendanceCharts(db)
    
    fig = charts.student_dashboard(index)
    charts.save(fig, f"{index}_dashboard")
    
    if not save_only:
        plt.show()
    plt.close(fig)


def show_all(save_only: bool = False) -> None:
    """Entry point for `infovis.py --all`."""
    db = Database()
    charts = AttendanceCharts(db)
    
    heatmap = charts.class_heatmap()
    charts.save(heatmap, "class_heatmap")
    
    totals = charts.sheet_totals()
    charts.save(totals, "sheet_totals")
    
    dist = charts.attendance_distribution()
    charts.save(dist, "attendance_distribution")
    
    if not save_only:
        # We only show the heatmap by default if not saving, as it's the most useful.
        plt.show()
    
    plt.close(heatmap)
    plt.close(totals)
    plt.close(dist)
