"""Tests for the present-or-absent decision (M7).

The rule these exercise is the one that decides whether a real student is
marked absent, so the cases here are deliberately the awkward ones: a cell just
under the threshold, a smudge with plenty of blobs but no stroke behind them,
and the lecturer's ruled-through ``ab``. Synthetic ``InkResult``s throughout —
the rule reads numbers, never pixels, so a test that loads a photo would be
slower and prove less.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from src import config
from src.detect.presence import decide, map_rows_to_students
from src.models import Cell, InkResult, Student


def make_ink(ink_ratio: float, components: int = 3, stroke_length: int = 300, row: int = 0) -> InkResult:
    """One measurement set, with signature-like defaults so each test varies one thing."""
    cell = Cell(row=row, col=config.SIGNATURE_COL, bbox=(0, 0, 220, 42))
    return InkResult(
        cell=cell,
        ink_ratio=ink_ratio,
        components=components,
        stroke_length=stroke_length,
        aspect=4.0,
    )


def test_clear_signature_is_present():
    """A well-covered cell with a long stroke is signed, and we are sure of it."""
    present, confidence = decide(make_ink(0.25))
    assert present is True
    assert confidence == pytest.approx(1.0)


def test_blank_cell_is_absent():
    """Near-zero ink with nothing in it is absent, with full confidence."""
    present, confidence = decide(make_ink(0.0, components=0, stroke_length=0))
    assert present is False
    assert confidence == pytest.approx(1.0)


def test_just_below_threshold_is_absent_with_low_confidence():
    """A cell sitting on the threshold is called absent — and flagged as a close call.

    Being unwilling to say 'I am not sure' is how a borderline cell gets
    silently mismarked, so confidence has to fall as the ink ratio approaches
    the threshold from below.
    """
    just_below = config.INK_RATIO_THRESHOLD * 0.99
    present, confidence = decide(make_ink(just_below, stroke_length=0))
    assert present is False
    assert confidence < config.UNCERTAIN_BELOW
    clearly_blank = decide(make_ink(0.0, components=0, stroke_length=0))[1]
    assert confidence < clearly_blank


def test_smudge_with_many_components_is_rejected():
    """Plenty of ink and plenty of blobs, but the pen never travelled: not a signature."""
    present, _ = decide(make_ink(0.20, components=12, stroke_length=5))
    assert present is False


def test_ink_that_is_not_a_signature_is_absent_but_uncertain():
    """The lecturer's ruled-through ``ab`` covers most of the cell.

    Ink coverage alone would call it present. The upper bound rejects it, and
    because the two conditions disagree the verdict is reported as the least
    certain one we make — which is what puts it in front of a human.
    """
    present, confidence = decide(make_ink(0.74, components=1, stroke_length=647))
    assert present is False
    assert confidence < config.UNCERTAIN_BELOW


def test_confidence_never_leaves_its_range():
    """Whatever goes in, confidence stays a probability-shaped number in [0.5, 1]."""
    for ink_ratio in (0.0, 0.001, config.INK_RATIO_THRESHOLD, 0.2, 0.9, 1.0):
        _, confidence = decide(make_ink(ink_ratio))
        assert 0.5 <= confidence <= 1.0


def test_sweep_rule_matches_decide():
    """``tools/tune_threshold.py`` must judge cells exactly as the pipeline does.

    The thresholds are chosen by sweeping a re-implementation of the rule. If
    the two ever drift apart, the numbers in the report would describe a system
    nobody is running — so this test pins them together.
    """
    from tools.tune_threshold import CellSample, predict

    cases = [
        (0.0, 0, 0),
        (0.0173, 4, 60),
        (0.0546, 6, 159),
        (0.25, 3, 300),
        (0.74, 1, 647),
        (config.INK_RATIO_THRESHOLD, 1, config.MIN_STROKE_LENGTH),
    ]
    for ink_ratio, components, stroke_length in cases:
        sample = CellSample(
            sheet_date="01.01.2019",
            student_index="10000409",
            truth=1,
            ink_ratio=ink_ratio,
            components=components,
            stroke_length=stroke_length,
            aspect=4.0,
            filled_ratio=0.0,
        )
        swept = predict(
            sample,
            config.INK_RATIO_THRESHOLD,
            min_stroke=config.MIN_STROKE_LENGTH,
            min_components=config.MIN_COMPONENTS,
            max_ink=config.MAX_INK_RATIO,
        )
        pipeline, _ = decide(make_ink(ink_ratio, components, stroke_length))
        assert swept == pipeline, f"rules disagree on ink={ink_ratio}"


def test_rows_map_to_students_by_position():
    """Row n is student n, and each student learns which row they are on."""
    cells = [Cell(row=row, col=config.SIGNATURE_COL, bbox=(0, 0, 10, 10)) for row in range(3)]
    students = [Student("10000409", "A"), Student("10009301", "B"), Student("10009302", "C")]

    mapping = map_rows_to_students(cells, students)

    assert mapping == {0: "10000409", 1: "10009301", 2: "10009302"}
    assert [student.row for student in students] == [0, 1, 2]


def test_row_count_mismatch_warns_and_maps_what_it_can(caplog):
    """A missing row must not shift every student up one.

    If M5 finds five rows where the roll has six, the safe failure is five
    correct records and a loud warning — not six records with five of them
    quietly attached to the wrong person.
    """
    cells = [Cell(row=row, col=config.SIGNATURE_COL, bbox=(0, 0, 10, 10)) for row in range(2)]
    students = [Student("10000409", "A"), Student("10009301", "B"), Student("10009302", "C")]

    with caplog.at_level("WARNING"):
        mapping = map_rows_to_students(cells, students)

    assert mapping == {0: "10000409", 1: "10009301"}
    assert "2 rows detected but 3 students" in caplog.text
