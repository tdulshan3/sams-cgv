#!/usr/bin/env python3
"""Run SAMS over all five sheets and report accuracy against ground truth."""

import subprocess
import os
import sys
import re
import pandas as pd
from pathlib import Path

# Adjust Python path if run directly from tools/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
from src.io.db import Database


def main():
    print("=" * 60)
    print(" SAMS End-to-End QA Runner")
    print("=" * 60)

    if config.DB_PATH.exists():
        print(f"Wiping existing database: {config.DB_PATH.name}")
        config.DB_PATH.unlink()

    sheets = sorted([p for p in config.SHEETS.iterdir() if p.is_file() and p.suffix == ".png"])
    if not sheets:
        print("No sheets found in data/sheets/")
        sys.exit(1)

    print(f"\nProcessing {len(sheets)} sheets...\n")

    for sheet in sheets:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, "sams.py", str(sheet), str(config.INFO_XML), "--no-show"]
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"[{sheet.name}] FAILED with exit code {result.returncode}")
            print(result.stderr)
            sys.exit(result.returncode)

        # Parse summary
        duration_match = re.search(r"Duration\s*:\s*([\d.]+)", result.stdout)
        students_match = re.search(r"Students\s*:\s*(\d+)", result.stdout)
        
        duration = duration_match.group(1) if duration_match else "?"
        students = students_match.group(1) if students_match else "?"
        
        print(f"[{sheet.name}] SUCCESS | {students} students | {duration} s")

    print("\n" + "=" * 60)
    print(" Accuracy Report")
    print("=" * 60)

    # Validate against ground truth
    if not config.GROUND_TRUTH.exists():
        print(f"Ground truth file missing: {config.GROUND_TRUTH}")
        sys.exit(1)

    gt_df = pd.read_csv(config.GROUND_TRUTH, dtype={'student_index': str})
    
    db = Database()
    records = db.get_all_attendance()
    if not records:
        print("Database is empty after processing.")
        sys.exit(1)

    db_df = pd.DataFrame(records)
    
    # Merge GT and DB results
    merged = gt_df.merge(db_df, on=['sheet_date', 'student_index'], suffixes=('_gt', '_db'))
    
    total = len(merged)
    correct = (merged['present_gt'] == merged['present_db']).sum()
    accuracy = correct / total * 100 if total > 0 else 0
    
    print(f"Overall Accuracy: {accuracy:.1f}% ({correct}/{total} cells correct)\n")

    # Per sheet accuracy
    print("Per Sheet Accuracy:")
    for sheet_date, group in merged.groupby('sheet_date'):
        sheet_total = len(group)
        sheet_correct = (group['present_gt'] == group['present_db']).sum()
        sheet_acc = sheet_correct / sheet_total * 100
        print(f"  {sheet_date}: {sheet_acc:.1f}% ({sheet_correct}/{sheet_total})")

    mistakes = merged[merged['present_gt'] != merged['present_db']]
    
    if not mistakes.empty:
        print("\nMisclassified Cells (Please review):")
        for _, row in mistakes.iterrows():
            gt_status = "Present" if row['present_gt'] else "Absent"
            db_status = "Present" if row['present_db'] else "Absent"
            note = f" (Note: {row['note']})" if pd.notna(row['note']) else ""
            print(f"  - {row['sheet_date']} / {row['student_index']}: "
                  f"Predicted {db_status}, should be {gt_status}{note}")
    else:
        print("\nPerfect Match! No misclassified cells.")

if __name__ == "__main__":
    main()
