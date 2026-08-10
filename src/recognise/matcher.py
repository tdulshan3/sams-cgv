"""Compare one student's signatures across sheets and report a mismatch.

This is the module ``investigate.py`` calls into, see that file's docstring
for the command line contract. Everything here assumes ``investigate.py`` (via
``src.cli.signature_samples``) has already confirmed there are at least two
saved signatures for the student; this module does not repeat that check on
the CLI path, but every function here is still safe to call directly (e.g.
from tests) with too few samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from collections.abc import Sequence
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src import config
from src.recognise import features
from src.recognise.preprocess_sig import normalise_signature
from src.utils.logging import get_logger

log = get_logger("recognise.matcher")

SHEET_DATE_FORMAT = "%d.%m.%Y"
"""Sheet folder names are e.g. ``12.07.2019``, day first, so a plain string
sort puts July before May. Every place that needs chronological order must
parse through this format rather than sorting the folder name directly.
"""


@dataclass
class SignatureSample:
    """One student's signature, as saved by M6 for a single sheet.

    ``mask`` is what every comparison in this module actually operates on,
    ink = 255, background = 0, not yet normalised. ``crop`` is kept alongside
    it purely for figures (T8's report images want to show the original
    colour ink, not just the mask).
    """

    student_index: str
    sheet_date: str
    mask: np.ndarray
    crop: np.ndarray | None = None


def _sheet_date_key(cell_dir: Path) -> tuple[int, str]:
    """Sort key for a sheet's cell folder: real chronological order first.

    Falls back to the raw name (pushed after every real date, via the ``1``
    tag) for any folder that doesn't parse; a stray or renamed folder should
    not crash a comparison run, just sort last and predictably.
    """
    try:
        return (0, datetime.strptime(cell_dir.name, SHEET_DATE_FORMAT).isoformat())
    except ValueError:
        log.warning("cell folder %s is not a %s date, sorting it last", cell_dir.name, SHEET_DATE_FORMAT)
        return (1, cell_dir.name)


def _read_mask(cell_dir: Path, index: str) -> np.ndarray | None:
    """Load the saved ink mask for one student in one sheet's cell folder.

    Returns ``None`` rather than raising when the mask file is absent; this
    is the situation before M6 lands, and callers decide what to do about it
    (see :func:`load_samples`, which falls back to deriving a rough mask from
    the colour crop so the pipeline stays runnable rather than blocking on
    M6's schedule).
    """
    mask_path = cell_dir / f"{index}_mask.png"
    if not mask_path.is_file():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        log.warning("could not decode mask %s", mask_path)
        return None
    return mask


def _bootstrap_mask(crop: np.ndarray) -> np.ndarray:
    """A rough ink mask from a colour crop, used only when M6's real mask is missing.

    Plain saturation-or-darkness threshold, deliberately not a real ink
    segmentation. This exists so ``investigate.py`` can be demonstrated end to
    end before M6's module merges; the moment a real ``<index>_mask.png`` is
    on disk, :func:`load_samples` prefers it and this function is never
    called for that sample. Not a substitute for M6's work and not meant to
    be tuned: it is a placeholder, in the same spirit as ``src/stubs.py``.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((saturation > 40) | (value < 150)).astype(np.uint8) * 255
    # Drop a thin border so a table line right at the crop edge isn't read as ink.
    border = 3
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def load_samples(index: str) -> list[SignatureSample]:
    """Load every saved signature for one student, oldest sheet first.

    Mirrors ``src.cli.signature_samples`` (same folder convention:
    ``outputs/cells/<sheet_date>/<index>.png``) but returns the actual pixel
    data rather than just the dates, since comparison needs the images.

    A student with zero saved crops gets an empty list back, not an
    exception: an empty or short list is an ordinary state of the project
    (see T1) and it is every caller's job to decide what to do about it, not
    this function's.
    """
    if not config.CELLS.is_dir():
        return []

    samples: list[SignatureSample] = []
    for cell_dir in sorted(config.CELLS.iterdir(), key=_sheet_date_key):
        if not cell_dir.is_dir():
            continue

        crop_path = cell_dir / f"{index}.png"
        if not crop_path.is_file():
            continue

        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if crop is None:
            log.warning("could not decode signature crop %s", crop_path)
            continue

        mask = _read_mask(cell_dir, index)
        if mask is None:
            log.debug(
                "no saved mask for %s on %s yet, using a rough bootstrap mask",
                index,
                cell_dir.name,
            )
            mask = _bootstrap_mask(crop)

        samples.append(
            SignatureSample(
                student_index=index,
                sheet_date=cell_dir.name,
                mask=mask,
                crop=crop,
            )
        )

    return samples


@dataclass
class MatchScore:
    """The result of comparing two signatures: five feature scores, one
    combined verdict.

    Every field except ``verdict`` is a float in ``[0, 1]`` where ``1`` means
    identical: the shared convention every function in ``features.py``
    guarantees. Keeping all five alongside the combined score (rather than
    just returning the combined float) is what lets T7's printout show the
    per-feature breakdown, not just a final number nobody can audit.
    """

    ssim: float
    hog: float
    hu: float
    orb: float
    custom: float
    combined: float
    verdict: str
    """One of ``"match"``, ``"mismatch"``, ``"uncertain"``, see :func:`_verdict`."""


@dataclass(frozen=True)
class PairComparison:
    """A scored pair of signature samples."""

    left: SignatureSample
    right: SignatureSample
    score: MatchScore


def short_sheet_date(sheet_date: str) -> str:
    """Return the short date label used in tables and figures."""
    try:
        return datetime.strptime(sheet_date, SHEET_DATE_FORMAT).strftime("%d.%m")
    except ValueError:
        return sheet_date


def discover_student_indices() -> list[str]:
    """Return every student index with at least one saved crop."""
    if not config.CELLS.is_dir():
        return []

    indices: set[str] = set()
    for cell_dir in config.CELLS.iterdir():
        if not cell_dir.is_dir():
            continue
        for crop_path in cell_dir.glob("*.png"):
            if crop_path.name.endswith("_mask.png"):
                continue
            indices.add(crop_path.stem)
    return sorted(indices)


def student_name(index: str) -> str:
    """Return the student's name from ``data/info.xml`` when available."""
    if not config.INFO_XML.is_file():
        return index

    try:
        tree = ET.parse(config.INFO_XML)
    except ET.ParseError:
        return index

    for student in tree.findall(".//student"):
        found_index = student.findtext("index")
        if found_index == index:
            name = student.findtext("name")
            return name.strip() if name else index
    return index


def load_all_samples() -> dict[str, list[SignatureSample]]:
    """Load all samples grouped by student index."""
    return {index: load_samples(index) for index in discover_student_indices()}


def pairwise_comparisons(samples: Sequence[SignatureSample]) -> list[PairComparison]:
    """Compare every unique pair of ``samples``."""
    comparisons: list[PairComparison] = []
    for left, right in combinations(samples, 2):
        comparisons.append(PairComparison(left=left, right=right, score=compare(left.mask, right.mask)))
    return comparisons


def compare_all_students(samples_by_student: dict[str, list[SignatureSample]]) -> tuple[list[PairComparison], list[PairComparison]]:
    """Build all genuine and impostor pair scores from the loaded samples."""
    genuine: list[PairComparison] = []
    impostor: list[PairComparison] = []

    for samples in samples_by_student.values():
        genuine.extend(pairwise_comparisons(samples))

    student_items = list(samples_by_student.items())
    for left_index, left_samples in student_items:
        for right_index, right_samples in student_items:
            if left_index >= right_index:
                continue
            for left_sample in left_samples:
                for right_sample in right_samples:
                    impostor.append(
                        PairComparison(
                            left=left_sample,
                            right=right_sample,
                            score=compare(left_sample.mask, right_sample.mask),
                        )
                    )

    return genuine, impostor


def combined_score_matrix(samples: Sequence[SignatureSample]) -> np.ndarray:
    """Return the symmetric matrix of combined similarity scores."""
    count = len(samples)
    matrix = np.zeros((count, count), dtype=np.float64)
    for row in range(count):
        matrix[row, row] = 1.0
        for column in range(row + 1, count):
            score = compare(samples[row].mask, samples[column].mask).combined
            matrix[row, column] = score
            matrix[column, row] = score
    return matrix


def mean_similarities(matrix: np.ndarray) -> np.ndarray:
    """Return the mean similarity of each row to all the other rows."""
    if matrix.size == 0:
        return np.array([], dtype=np.float64)
    if matrix.shape[0] == 1:
        return np.array([1.0], dtype=np.float64)

    totals = matrix.sum(axis=1) - np.diag(matrix)
    return totals / (matrix.shape[0] - 1)


def flagged_outlier(samples: Sequence[SignatureSample]) -> tuple[int, np.ndarray] | None:
    """Return the least similar sample and all per-sample mean similarities."""
    if len(samples) < 2:
        return None
    matrix = combined_score_matrix(samples)
    means = mean_similarities(matrix)
    return int(np.argmin(means)), means


def feature_separation(genuine: Sequence[PairComparison], impostor: Sequence[PairComparison]) -> dict[str, float]:
    """Measure how far apart genuine and impostor averages are per feature."""
    features_to_measure = ("ssim", "hog", "hu", "orb", "custom", "combined")
    separation: dict[str, float] = {}

    for name in features_to_measure:
        genuine_scores = np.array([getattr(pair.score, name) for pair in genuine], dtype=np.float64)
        impostor_scores = np.array([getattr(pair.score, name) for pair in impostor], dtype=np.float64)
        if genuine_scores.size == 0 or impostor_scores.size == 0:
            separation[name] = 0.0
            continue
        separation[name] = float(genuine_scores.mean() - impostor_scores.mean())

    return separation


def threshold_curve(genuine_scores: Sequence[float], impostor_scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute FAR and FRR across thresholds and return the EER point."""
    genuine_array = np.asarray(list(genuine_scores), dtype=np.float64)
    impostor_array = np.asarray(list(impostor_scores), dtype=np.float64)

    if genuine_array.size == 0 or impostor_array.size == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, float("nan"), float("nan")

    thresholds = np.unique(np.concatenate(([0.0], genuine_array, impostor_array, [1.0])))
    fars = np.empty_like(thresholds, dtype=np.float64)
    frrs = np.empty_like(thresholds, dtype=np.float64)

    for index, threshold in enumerate(thresholds):
        fars[index] = float(np.mean(impostor_array >= threshold))
        frrs[index] = float(np.mean(genuine_array < threshold))

    differences = np.abs(fars - frrs)
    eer_index = int(np.argmin(differences))
    eer_threshold = float(thresholds[eer_index])
    eer_rate = float((fars[eer_index] + frrs[eer_index]) / 2.0)
    return thresholds, fars, frrs, eer_threshold, eer_rate


def _save_figure(figure: plt.Figure, path: Path) -> None:
    """Save a figure at the project dpi and close it immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_similarity_matrix_figure(samples: Sequence[SignatureSample], out_path: Path, *, title: str | None = None) -> tuple[int, np.ndarray] | None:
    """Save a heatmap of combined pairwise similarities for one student."""
    if len(samples) < 2:
        return None

    matrix = combined_score_matrix(samples)
    means = mean_similarities(matrix)
    figure, axis = plt.subplots(figsize=(max(4.5, len(samples) * 1.2), max(4.0, len(samples) * 1.0)))
    heatmap = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(samples)))
    axis.set_yticks(range(len(samples)))
    axis.set_xticklabels([short_sheet_date(sample.sheet_date) for sample in samples], rotation=45, ha="right")
    axis.set_yticklabels([short_sheet_date(sample.sheet_date) for sample in samples])
    axis.set_xlabel("sheet date")
    axis.set_ylabel("sheet date")
    if title is not None:
        axis.set_title(title)
    figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04, label="combined score")
    for row in range(len(samples)):
        for column in range(len(samples)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] < 0.55 else "black",
                fontsize=8,
            )
    _save_figure(figure, out_path)
    return flagged_outlier(samples)


def save_feature_separation_figure(separation: dict[str, float], out_path: Path) -> None:
    """Plot the genuine-vs-impostor separation for each feature."""
    features_order = ["ssim", "hog", "hu", "orb", "custom", "combined"]
    values = [separation.get(name, 0.0) for name in features_order]

    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    bars = axis.bar(features_order, values, color=["#3b82f6" if value >= 0 else "#ef4444" for value in values])
    axis.axhline(0.0, color="#222", linewidth=0.8)
    axis.set_ylabel("mean(genuine) - mean(impostor)")
    axis.set_title("Feature separation")
    for bar, value in zip(bars, values, strict=False):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    _save_figure(figure, out_path)


def save_threshold_curve_figure(thresholds: np.ndarray, fars: np.ndarray, frrs: np.ndarray, eer_threshold: float, eer_rate: float, out_path: Path) -> None:
    """Plot FAR and FRR as the combined threshold sweeps from low to high."""
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.plot(thresholds, fars, label="FAR", color="#ef4444")
    axis.plot(thresholds, frrs, label="FRR", color="#2563eb")
    axis.scatter([eer_threshold], [eer_rate], color="#111", zorder=5, label=f"EER {eer_rate:.3f} @ {eer_threshold:.3f}")
    axis.set_xlabel("combined threshold")
    axis.set_ylabel("rate")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("FAR / FRR threshold sweep")
    axis.legend()
    _save_figure(figure, out_path)


def save_pair_distribution_figure(genuine_scores: Sequence[float], impostor_scores: Sequence[float], out_path: Path) -> None:
    """Plot the genuine and impostor combined-score distributions together."""
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0.0, 1.0, 26)
    axis.hist(genuine_scores, bins=bins, alpha=0.65, label="genuine", color="#2563eb")
    axis.hist(impostor_scores, bins=bins, alpha=0.65, label="impostor", color="#ef4444")
    axis.set_xlabel("combined score")
    axis.set_ylabel("pair count")
    axis.set_title("Signature score distributions")
    axis.legend()
    _save_figure(figure, out_path)


def _format_pair_row(left: SignatureSample, right: SignatureSample, score: MatchScore) -> str:
    """Format one row of the investigate score table."""
    pair_label = f"{short_sheet_date(left.sheet_date)} vs {short_sheet_date(right.sheet_date)}"
    return (
        f"  {pair_label:<27} "
        f"{score.ssim:>5.2f}   {score.hog:>5.2f}   {score.hu:>5.2f}   {score.orb:>5.2f}   {score.custom:>5.2f}   {score.combined:>7.2f}  {score.verdict}"
    )


def _investigate_summary(samples: Sequence[SignatureSample], comparisons: Sequence[PairComparison]) -> str:
    """Build the final verdict line for :func:`investigate`."""
    if not comparisons:
        return "Verdict: all samples match."

    if all(pair.score.verdict == "match" for pair in comparisons):
        return "Verdict: all samples match."

    outlier = flagged_outlier(samples)
    if outlier is None:
        return "Verdict: all samples match."

    outlier_index, means = outlier
    outlier_sample = samples[outlier_index]
    other_means = np.delete(means, outlier_index)
    other_mean = float(other_means.mean()) if other_means.size else float(means[outlier_index])
    return (
        f"Verdict: signature on {outlier_sample.sheet_date} does not match the others "
        f"(mean {means[outlier_index]:.2f} vs {other_mean:.2f})."
    )


def _weighted_combine(ssim: float, hog: float, hu: float, orb: float, custom: float) -> float:
    """Combine the five feature scores into one, using ``config.SCORE_WEIGHTS``.

    A plain weighted sum, not an average of some cleverer kind; every
    weight is a fraction of one whole (they sum to 1.0, see the config
    docstring), so the result stays in ``[0, 1]`` automatically as long as
    every input does. Which weights are right is not decided here: T5's
    genuine-vs-impostor experiment is what justifies (or revises) the values
    in ``config.SCORE_WEIGHTS``, this function just applies whatever they are.
    """
    weights = config.SCORE_WEIGHTS
    return (
        weights["ssim"] * ssim
        + weights["hog"] * hog
        + weights["hu"] * hu
        + weights["orb"] * orb
        + weights["custom"] * custom
    )


def _verdict(combined: float) -> str:
    """Turn a combined score into a match / mismatch / uncertain verdict.

    Three-way rather than a plain yes/no on purpose: ``config.MATCH_THRESHOLD``
    was set from an experiment on 41 genuine pairs (T5), not a law of nature,
    so a score that lands close to it either side is exactly the kind of case
    the threshold itself is least sure about. Reporting that honestly as
    "uncertain", rather than forcing a confident-looking match or mismatch
    out of a borderline number, is, per the brief, a strength of this
    module, not a weakness to hide.

    * ``combined >= MATCH_THRESHOLD`` → ``"match"``
    * ``combined <= MATCH_THRESHOLD - UNCERTAIN_BAND`` → ``"mismatch"``
    * anything in between → ``"uncertain"``
    """
    if combined >= config.MATCH_THRESHOLD:
        return "match"
    if combined <= config.MATCH_THRESHOLD - config.UNCERTAIN_BAND:
        return "mismatch"
    return "uncertain"


def compare(a: np.ndarray, b: np.ndarray) -> MatchScore:
    """Compare two raw signature ink masks end to end.

    Parameters
    ----------
    a, b:
        Ink masks, ink = 255, background = 0, *not* yet normalised.
        ``compare`` does that itself, so every caller (T1's loader, T6's
        similarity matrix, T5's eval script) can pass a
        :class:`SignatureSample`'s raw ``mask`` straight through without
        remembering to normalise it first, and every comparison in the
        project runs through the exact same normalisation step.

    Returns
    -------
    A :class:`MatchScore` carrying all five individual feature scores, the
    combined weighted score, and a match/mismatch/uncertain verdict.
    """
    norm_a = normalise_signature(a, config.SIG_NORM_SIZE)
    norm_b = normalise_signature(b, config.SIG_NORM_SIZE)

    ssim = features.ssim_similarity(norm_a, norm_b)
    hog = features.hog_similarity(norm_a, norm_b)
    hu = features.hu_similarity(norm_a, norm_b)
    orb = features.orb_similarity(norm_a, norm_b)
    custom = features.custom_similarity(norm_a, norm_b)

    combined = _weighted_combine(ssim, hog, hu, orb, custom)

    return MatchScore(
        ssim=ssim,
        hog=hog,
        hu=hu,
        orb=orb,
        custom=custom,
        combined=combined,
        verdict=_verdict(combined),
    )


MIN_SAMPLES = 2
"""One signature cannot be compared with anything, mirrors investigate.py's
own constant. Duplicated rather than imported so this module has no
dependency on the CLI layer; ``investigate.py`` already screens this case
before calling in, but ``investigate()`` must still handle it safely on its
own for direct callers such as the test suite.
"""


def investigate(index: str, save_only: bool = False) -> None:
    """Compare every pair of a student's signatures and report a mismatch.

    This is the function ``investigate.py`` calls. Parameters
    ----------
    index:
        Student index, e.g. ``"10000409"``.
    save_only:
        When ``True``, write the comparison figure to
        ``outputs/figures/`` and print no window, for headless / CI runs.

    A student with fewer than :data:`MIN_SAMPLES` saved signatures is not an
    error: there is nothing to compare yet, so this prints a plain
    explanation and returns rather than raising.
    """
    samples = load_samples(index)
    if len(samples) < MIN_SAMPLES:
        print(
            f"student {index} has {len(samples)} saved signature "
            f"{'sample' if len(samples) == 1 else 'samples'}, "
            f"at least {MIN_SAMPLES} are needed to compare"
        )
        return

    comparisons = pairwise_comparisons(samples)

    print(f"Student {index}, {student_name(index)}, {len(samples)} samples")
    print("  pair                        SSIM   HOG    HU     ORB    OWN    COMBINED")
    for comparison in comparisons:
        print(_format_pair_row(comparison.left, comparison.right, comparison.score))

    print(_investigate_summary(samples, comparisons))

    matrix = combined_score_matrix(samples)
    means = mean_similarities(matrix)
    outlier_index = int(np.argmin(means)) if means.size else -1

    figure_path = config.FIGURES / f"m8_investigate_{index}.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(samples), figsize=(max(6.0, 2.4 * len(samples)), 3.8), squeeze=False)
    axes_row = axes[0]
    for position, (axis, sample, mean_score) in enumerate(zip(axes_row, samples, means, strict=False)):
        axis.imshow(sample.crop if sample.crop is not None else sample.mask, cmap=None if sample.crop is not None else "gray")
        axis.set_title(f"{short_sheet_date(sample.sheet_date)}\nmean {mean_score:.2f}", fontsize=9)
        axis.axis("off")
        if position == outlier_index:
            for spine in axis.spines.values():
                spine.set_edgecolor("#ef4444")
                spine.set_linewidth(3.0)

    figure.suptitle(f"Student {index} signature comparison", fontsize=12)
    figure.tight_layout()
    _save_figure(figure, figure_path)

    if not save_only:
        plt.show()