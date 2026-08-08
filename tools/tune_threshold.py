#!/usr/bin/env python3
"""Choose the decision thresholds by measuring them, not by feel.

``INK_RATIO_THRESHOLD`` decides who is marked absent, so it is the one number
in M7 that must not be guessed. This runs the real pipeline over all five
sheets, joins what M6 measured in each of the thirty signature cells against
the hand-labelled ``data/ground_truth.csv``, and sweeps the threshold across
its whole range to show what each value would score::

    python tools/tune_threshold.py              # run the pipeline, then sweep
    python tools/tune_threshold.py --cached     # re-sweep the last run's features

The pipeline pass takes about ten seconds a sheet, so the features are cached
at ``outputs/decision_features.csv`` and ``tools/make_m7_figures.py`` draws its
figures from that same file — the numbers in the report and the numbers behind
the chosen threshold are then guaranteed to be the same numbers.

Accuracy is always reported twice: over all 30 cells, and over the 28 that
exclude the two cells known in advance to hold ink that is not a signature
(BUILD_SPEC.md §4 deviation 7). Quoting only the second would be flattering
and dishonest; quoting only the first hides which failures are the decision
rule's fault and which are the ink mask's.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src import config  # noqa: E402
from src.io.xml_parser import parse_info  # noqa: E402
from src.models import SheetMeta, Student  # noqa: E402
from src.pipeline import Pipeline  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("tune")

FEATURES_CSV = config.OUTPUTS / "decision_features.csv"
"""Where one pipeline pass over all five sheets is cached."""

KNOWN_INK_BUT_ABSENT = {
    ("21.06.2019", "10009306"),
    ("05.07.2019", "10009303"),
}
"""The two cells that hold ink meaning *absent*, named in the spec before any
code was written. Accuracy is reported both with and without them."""

SWEEP_POINTS = 400
"""Steps across the ink-ratio range. Fine enough that the plateau edges land
where the data actually puts them rather than on a round number."""


@dataclass
class CellSample:
    """One signature cell: what M6 measured, and what a human says is true."""

    sheet_date: str
    student_index: str
    truth: int
    ink_ratio: float
    components: int
    stroke_length: int
    aspect: float
    filled_ratio: float
    note: str = ""

    @property
    def is_known_case(self) -> bool:
        """One of the two ink-but-absent cells."""
        return (self.sheet_date, self.student_index) in KNOWN_INK_BUT_ABSENT


def load_ground_truth() -> dict[tuple[str, str], tuple[int, str]]:
    """``{(date, index): (present, note)}`` from the hand-labelled CSV."""
    if not config.GROUND_TRUTH.is_file():
        raise FileNotFoundError(
            f"No ground truth at {config.GROUND_TRUTH}. It is committed — check your clone."
        )
    truth: dict[tuple[str, str], tuple[int, str]] = {}
    with config.GROUND_TRUTH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["sheet_date"], row["student_index"])
            truth[key] = (int(row["present"]), row.get("note", ""))
    return truth


def collect_features() -> list[CellSample]:
    """Run the pipeline over every sheet and pair each cell with its label.

    The decision stage is deliberately *not* in this pipeline. Tuning has to
    see M6's raw measurements, not a verdict already made with the thresholds
    being tuned.
    """
    from src.detect.ink_mask import InkStage
    from src.preprocess.binarize import BinarizeStage
    from src.preprocess.deskew import GeometryStage
    from src.preprocess.enhance import EnhanceStage
    from src.table.cell_extract import TableStage

    students, meta = parse_info()
    truth = load_ground_truth()
    samples: list[CellSample] = []

    for path in sorted(config.SHEETS.glob("*.png")):
        sheet = SheetMeta(path=path, date=path.stem, subject_code=meta["subject_code"])
        stages = [GeometryStage(), EnhanceStage(), BinarizeStage(), TableStage(), InkStage()]
        roll = [Student(student.index, student.name) for student in students]
        ctx = Pipeline(stages, viewer=None).run(sheet, roll)

        ink = ctx.get("ink") or []
        if len(ink) != len(students):
            log.warning(
                "%s: %d cells for %d students — pairing the first %d",
                path.stem,
                len(ink),
                len(students),
                min(len(ink), len(students)),
            )
        for result, student in zip(ink, students):
            present, note = truth.get((path.stem, student.index), (-1, ""))
            if present < 0:
                log.warning("no ground truth for %s on %s", student.index, path.stem)
                continue
            samples.append(
                CellSample(
                    sheet_date=path.stem,
                    student_index=student.index,
                    truth=present,
                    ink_ratio=float(result.ink_ratio),
                    components=int(result.components),
                    stroke_length=int(result.stroke_length),
                    aspect=float(result.aspect),
                    filled_ratio=0.0,
                    note=note,
                )
            )
    return samples


def save_features(samples: list[CellSample], path: Path = FEATURES_CSV) -> None:
    """Cache one pipeline pass so the figures need not repeat it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [field.name for field in fields(CellSample)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for sample in samples:
            writer.writerow({name: getattr(sample, name) for name in names})
    log.info("%d cells cached at %s", len(samples), path)


def load_features(path: Path = FEATURES_CSV) -> list[CellSample]:
    """Read the cache back, or say clearly that there is not one yet."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No cached features at {path} — run `python tools/tune_threshold.py` first"
        )
    samples = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            samples.append(
                CellSample(
                    sheet_date=row["sheet_date"],
                    student_index=row["student_index"],
                    truth=int(row["truth"]),
                    ink_ratio=float(row["ink_ratio"]),
                    components=int(row["components"]),
                    stroke_length=int(row["stroke_length"]),
                    aspect=float(row["aspect"]),
                    filled_ratio=float(row["filled_ratio"]),
                    note=row.get("note", ""),
                )
            )
    return samples


def predict(
    sample: CellSample,
    threshold: float,
    min_stroke: int = config.MIN_STROKE_LENGTH,
    min_components: int = config.MIN_COMPONENTS,
    max_ink: float = 1.0,
) -> bool:
    """The decision rule, with its thresholds passed in so they can be swept.

    Kept deliberately in step with :func:`src.detect.presence.decide`; the
    sweep would be worthless if it measured a different rule from the one the
    pipeline runs. ``tests/test_decision.py`` asserts the two agree.
    """
    return (
        threshold <= sample.ink_ratio <= max_ink
        and sample.components >= min_components
        and sample.stroke_length >= min_stroke
    )


def confusion(samples: list[CellSample], threshold: float, **rule) -> tuple[int, int, int, int]:
    """``(true_present, false_present, true_absent, false_absent)``."""
    tp = fp = tn = fn = 0
    for sample in samples:
        predicted = predict(sample, threshold, **rule)
        if predicted and sample.truth == 1:
            tp += 1
        elif predicted and sample.truth == 0:
            fp += 1
        elif not predicted and sample.truth == 0:
            tn += 1
        else:
            fn += 1
    return tp, fp, tn, fn


def accuracy(samples: list[CellSample], threshold: float, **rule) -> float:
    """Fraction of cells the rule gets right at this threshold."""
    if not samples:
        return 0.0
    tp, fp, tn, fn = confusion(samples, threshold, **rule)
    return (tp + tn) / len(samples)


def sweep(samples: list[CellSample], **rule) -> tuple[list[float], list[float]]:
    """Accuracy at every threshold from 0 to just past the largest ink ratio."""
    top = max((sample.ink_ratio for sample in samples), default=0.1) * 1.05
    thresholds = [top * step / SWEEP_POINTS for step in range(SWEEP_POINTS + 1)]
    return thresholds, [accuracy(samples, value, **rule) for value in thresholds]


def best_plateau(thresholds: list[float], scores: list[float]) -> tuple[float, float, float]:
    """The widest run of top-scoring thresholds, and its midpoint.

    A sweep over thirty cells does not have a peak, it has a plateau: every
    threshold that falls in the gap between the highest absent cell and the
    lowest present cell scores identically. Picking the first such value would
    put the threshold hard against one cell's measurement. The midpoint of the
    widest plateau is the value furthest from being wrong about any cell.

    Returns:
        ``(chosen, low, high)`` — the midpoint and the plateau's two edges.
    """
    top = max(scores)
    best_run: tuple[int, int] | None = None
    start: int | None = None
    for position, score in enumerate([*scores, -1.0]):
        if score == top and start is None:
            start = position
        elif score != top and start is not None:
            if best_run is None or position - start > best_run[1] - best_run[0]:
                best_run = (start, position)
            start = None
    if best_run is None:
        return thresholds[0], thresholds[0], thresholds[0]
    low, high = thresholds[best_run[0]], thresholds[best_run[1] - 1]
    return (low + high) / 2, low, high


def _report(title: str, samples: list[CellSample], threshold: float, **rule) -> None:
    """Print one accuracy block, and name every cell the rule gets wrong."""
    tp, fp, tn, fn = confusion(samples, threshold, **rule)
    total = len(samples)
    correct = tp + tn
    print(f"\n{title}")
    print(f"  cells            : {total}")
    print(f"  accuracy         : {correct}/{total} = {correct / total:.1%}")
    print(f"  present correct  : {tp}   absent correct : {tn}")
    print(f"  false present    : {fp}   false absent   : {fn}")
    wrong = [
        sample
        for sample in samples
        if predict(sample, threshold, **rule) != bool(sample.truth)
    ]
    for sample in wrong:
        called = "present" if predict(sample, threshold, **rule) else "absent"
        print(
            f"    wrong: {sample.sheet_date} {sample.student_index} "
            f"called {called}, ink={sample.ink_ratio:.4f} stroke={sample.stroke_length}"
            + (f"  [{sample.note}]" if sample.note else "")
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tune_threshold.py",
        description="sweep the ink ratio threshold against the hand-labelled ground truth",
    )
    parser.add_argument(
        "--cached", action="store_true", help="re-use the last run's features instead of processing"
    )
    args = parser.parse_args(argv)

    samples = load_features() if args.cached else collect_features()
    if not args.cached:
        save_features(samples)

    present = [sample for sample in samples if sample.truth == 1]
    absent = [sample for sample in samples if sample.truth == 0]
    print("\nink ratio by label")
    print(
        f"  present ({len(present)}): "
        f"{min(s.ink_ratio for s in present):.4f} … {max(s.ink_ratio for s in present):.4f}"
    )
    print(
        f"  absent  ({len(absent)}): "
        f"{min(s.ink_ratio for s in absent):.4f} … {max(s.ink_ratio for s in absent):.4f}"
    )

    # Two sweeps, because they answer two different questions. The threshold is
    # chosen from the first: what can ink coverage do on its own? Read off the
    # full rule instead and the plateau runs all the way down to zero — not
    # because a threshold of zero is sound, but because MIN_STROKE_LENGTH is
    # quietly covering for it. A threshold picked there would be resting on
    # another feature and would fail the moment that feature moved.
    ink_only = dict(min_stroke=0, min_components=0, max_ink=1.0)
    full_rule = dict(
        min_stroke=config.MIN_STROKE_LENGTH,
        min_components=config.MIN_COMPONENTS,
        max_ink=config.MAX_INK_RATIO,
    )
    thresholds, ink_scores = sweep(samples, **ink_only)
    chosen, low, high = best_plateau(thresholds, ink_scores)
    _, rule_scores = sweep(samples, **full_rule)

    print("\nsweep — ink ratio alone (chooses the threshold)")
    print(f"  best accuracy    : {max(ink_scores):.1%}")
    print(f"  plateau          : {low:.4f} … {high:.4f}")
    print(f"  chosen (midpoint): {chosen:.4f}")
    print(f"  config currently : {config.INK_RATIO_THRESHOLD:.4f}")
    print("\nsweep — full rule (what the extra features buy)")
    print(f"  best accuracy    : {max(rule_scores):.1%}")
    print(f"  at chosen value  : {accuracy(samples, config.INK_RATIO_THRESHOLD, **full_rule):.1%}")

    rule = full_rule

    _report("all 30 cells", samples, config.INK_RATIO_THRESHOLD, **rule)
    _report(
        "excluding the two ink-but-absent cells",
        [sample for sample in samples if not sample.is_known_case],
        config.INK_RATIO_THRESHOLD,
        **rule,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
