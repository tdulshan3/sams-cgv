"""Investigation report for one student's signature run.

``matcher.investigate`` assembles a report object and returns it so callers
can inspect the results programmatically, tests, the evaluation tool, and
any future visualisation layer, rather than only reading printed output.

The report carries the same information ``investigate.py`` prints, in a form
that is easy to pass around and inspect without capturing stdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.recognise.matcher import PairComparison, SignatureSample

import numpy as np


@dataclass
class InvestigationReport:
    """Everything produced by a single call to :func:`matcher.investigate`.

    Attributes
    ----------
    student_index:
        The index that was investigated, e.g. ``"10000409"``.
    student_name:
        Display name from ``info.xml``, or the index when the XML is absent.
    samples:
        All signature samples loaded for this student, oldest-first.
    comparisons:
        Every unique pair of samples with their individual and combined scores.
    similarity_matrix:
        Symmetric ``(n, n)`` matrix of combined scores for the ``n`` samples.
    mean_similarities:
        Mean combined score of each sample against the rest.
    outlier_index:
        Index into ``samples`` of the least similar sample, or ``-1`` when
        there are fewer than two samples (nothing to be an outlier against).
    verdict:
        The text verdict line, e.g.
        ``"Verdict: signature on 31.07.2019 does not match the others."``.
    figure_path:
        Path to the saved ``m8_investigate_<index>.png``, as a string, or an
        empty string when the figure was not saved (``save_only=False`` path).
    """

    student_index: str
    student_name: str
    samples: list[SignatureSample] = field(default_factory=list)
    comparisons: list[PairComparison] = field(default_factory=list)
    similarity_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    mean_similarities: np.ndarray = field(default_factory=lambda: np.empty(0))
    outlier_index: int = -1
    verdict: str = ""
    figure_path: str = ""

    @property
    def has_enough_samples(self) -> bool:
        """``True`` when at least two samples were available to compare."""
        return len(self.samples) >= 2

    @property
    def outlier_sample(self) -> SignatureSample | None:
        """The flagged sample, or ``None`` when there is no outlier."""
        if self.outlier_index < 0 or self.outlier_index >= len(self.samples):
            return None
        return self.samples[self.outlier_index]
