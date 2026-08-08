"""Write all M8 signature-recognition figures to ``outputs/figures/``.

Follows the same convention as ``make_m6_figures.py`` and
``make_m7_figures.py``: run from the repository root, write the figures,
print a summary. The heavy lifting delegates to
:func:`tools.eval_recognition.build_evaluation` so this file stays thin.

Usage::

    python tools/make_m8_figures.py
    python tools/make_m8_figures.py --show   # open figures after saving
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Scripts in tools/ run as ``python tools/<name>.py`` from the repository
# root. Insert the root so the ``src`` package resolves the same way it does
# everywhere else.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.eval_recognition import build_evaluation, load_ground_truth_summary  # noqa: E402


def main() -> None:
    """Entry point: build evaluation, report results, exit."""
    parser = argparse.ArgumentParser(
        description="Write the M8 signature-recognition figures to outputs/figures/."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open each figure in a window after saving it",
    )
    args = parser.parse_args()

    summary = build_evaluation(save_only=not args.show)
    ground_truth = load_ground_truth_summary()

    print()
    if summary.genuine:
        print(
            f"Ground truth: {ground_truth.genuine_pair_count} genuine pairs "
            f"across {len(ground_truth.present_counts)} students"
        )
        print(
            f"EER threshold: {summary.eer_threshold:.3f}  "
            f"(EER {summary.eer_rate:.3f})"
        )
        if summary.best_feature:
            print(f"Best separating feature: {summary.best_feature}")
        print()
        print("Figures written to outputs/figures/:")
        figures = [
            "m8_normalisation_steps.png",
            "m8_score_distributions.png",
            "m8_far_frr_curve.png",
            "m8_similarity_matrix.png",
            "m8_orb_matches.png",
            "m8_feature_comparison.png",
            "m8_flagged_example.png",
        ]
        for name in figures:
            print(f"  {name}")
    else:
        print(
            "No crops found under outputs/cells/ — placeholder figures written.\n"
            "Run  python sams.py <sheet> data/info.xml  first to generate crops."
        )


if __name__ == "__main__":
    main()
