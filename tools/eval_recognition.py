"""Evaluate signature recognition quality for M8.

This script builds the genuine and impostor pair sets from the saved crops
under ``outputs/cells/`` and writes the M8 figures in ``outputs/figures/``.
It is intentionally data-driven: if the crop tree is empty, it reports that
state clearly rather than fabricating scores.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src import config
from src.recognise import features, matcher
from src.recognise.preprocess_sig import centre_by_mass, normalise_signature, resize_keep_aspect, trim_to_ink


@dataclass(frozen=True)
class GroundTruthSummary:
    """Counts derived from ``data/ground_truth.csv``."""

    present_counts: dict[str, int]
    genuine_pair_count: int


@dataclass(frozen=True)
class EvaluationSummary:
    """All headline outputs from the recognition experiment."""

    genuine: list[matcher.PairComparison]
    impostor: list[matcher.PairComparison]
    thresholds: np.ndarray
    fars: np.ndarray
    frrs: np.ndarray
    eer_threshold: float
    eer_rate: float
    separation: dict[str, float]
    best_feature: str


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def _placeholder_figure(title: str, message: str, out_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.6, title, ha="center", va="center", fontsize=15, weight="bold")
    axis.text(0.5, 0.4, message, ha="center", va="center", fontsize=11)
    _save_figure(figure, out_path)


def load_ground_truth_summary() -> GroundTruthSummary:
    """Count present samples and genuine pairs from the CSV ground truth."""
    counts: dict[str, int] = {}
    if not config.GROUND_TRUTH.is_file():
        return GroundTruthSummary(present_counts=counts, genuine_pair_count=0)

    with config.GROUND_TRUTH.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            if row.get("present") == "1":
                index = row["student_index"]
                counts[index] = counts.get(index, 0) + 1

    genuine_pairs = sum(count * (count - 1) // 2 for count in counts.values() if count >= 2)
    return GroundTruthSummary(present_counts=counts, genuine_pair_count=genuine_pairs)


def _build_norm_step_images(sample: matcher.SignatureSample) -> list[tuple[str, np.ndarray, str | None]]:
    raw_crop = sample.crop if sample.crop is not None else cv2.cvtColor(sample.mask, cv2.COLOR_GRAY2BGR)
    raw_mask = sample.mask
    trimmed, _ = trim_to_ink(raw_mask, pad=config.SIG_NORM_PAD)
    scaled = resize_keep_aspect(trimmed, config.SIG_NORM_SIZE)
    centred = centre_by_mass(scaled, config.SIG_NORM_SIZE)
    normalised = normalise_signature(raw_mask)
    return [
        ("raw crop", raw_crop, None if raw_crop.ndim == 3 else "gray"),
        ("raw mask", raw_mask, "gray"),
        ("trimmed", trimmed, "gray"),
        ("scaled", scaled, "gray"),
        ("centred", centred, "gray"),
        ("normalised", normalised, "gray"),
    ]


def save_normalisation_steps(sample: matcher.SignatureSample, out_path: Path) -> None:
    items = _build_norm_step_images(sample)
    figure, axes = plt.subplots(2, 3, figsize=(10.0, 6.5))
    flat_axes = axes.ravel()
    for axis, (title, image, cmap) in zip(flat_axes, items, strict=False):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Normalisation steps for {sample.student_index} on {sample.sheet_date}")
    figure.tight_layout()
    _save_figure(figure, out_path)


def _good_orb_matches(norm_a: np.ndarray, norm_b: np.ndarray) -> tuple[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch], np.ndarray | None, np.ndarray | None]:
    keypoints_a, descriptors_a = features.orb_keypoints(norm_a)
    keypoints_b, descriptors_b = features.orb_keypoints(norm_b)
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) == 0 or len(descriptors_b) == 0:
        return keypoints_a, keypoints_b, [], descriptors_a, descriptors_b

    matcher_obj = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher_obj.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches: list[cv2.DMatch] = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < config.ORB_LOWE_RATIO * second.distance:
            good_matches.append(best)
    return keypoints_a, keypoints_b, good_matches, descriptors_a, descriptors_b


def save_orb_matches(left: matcher.SignatureSample, right: matcher.SignatureSample, out_path: Path) -> None:
    norm_left = normalise_signature(left.mask)
    norm_right = normalise_signature(right.mask)
    keypoints_left, keypoints_right, good_matches, _, _ = _good_orb_matches(norm_left, norm_right)

    if good_matches:
        draw = cv2.drawMatches(
            cv2.cvtColor(norm_left, cv2.COLOR_GRAY2BGR),
            keypoints_left,
            cv2.cvtColor(norm_right, cv2.COLOR_GRAY2BGR),
            keypoints_right,
            good_matches[:60],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        image = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        title = f"ORB matches: {left.sheet_date} vs {right.sheet_date} ({len(good_matches)} good matches)"
    else:
        left_rgb = cv2.cvtColor(norm_left, cv2.COLOR_GRAY2RGB)
        right_rgb = cv2.cvtColor(norm_right, cv2.COLOR_GRAY2RGB)
        image = np.concatenate([left_rgb, right_rgb], axis=1)
        title = f"ORB matches: {left.sheet_date} vs {right.sheet_date} (no confident matches)"

    figure, axis = plt.subplots(figsize=(10.0, 4.0))
    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")
    _save_figure(figure, out_path)


def save_flagged_example(samples: list[matcher.SignatureSample], out_path: Path) -> None:
    if len(samples) < 2:
        _placeholder_figure("Flagged example", "Need at least two samples to build an outlier figure.", out_path)
        return

    matrix = matcher.combined_score_matrix(samples)
    means = matcher.mean_similarities(matrix)
    outlier_index = int(np.argmin(means))

    figure, axes = plt.subplots(1, len(samples), figsize=(max(6.0, 2.4 * len(samples)), 3.8), squeeze=False)
    axes_row = axes[0]
    for position, (axis, sample, mean_score) in enumerate(zip(axes_row, samples, means, strict=False)):
        axis.imshow(sample.crop if sample.crop is not None else sample.mask, cmap=None if sample.crop is not None else "gray")
        axis.set_title(f"{matcher.short_sheet_date(sample.sheet_date)}\nmean {mean_score:.2f}", fontsize=9)
        axis.axis("off")
        if position == outlier_index:
            for spine in axis.spines.values():
                spine.set_edgecolor("#ef4444")
                spine.set_linewidth(3.0)

    figure.suptitle(f"Outlier candidate: {samples[outlier_index].sheet_date}")
    figure.tight_layout()
    _save_figure(figure, out_path)


def build_evaluation(save_only: bool = True) -> EvaluationSummary:
    """Run the genuine/impostor experiment and save every headline figure."""
    config.ensure_dirs()

    samples_by_student = matcher.load_all_samples()
    genuine, impostor = matcher.compare_all_students(samples_by_student)

    if not samples_by_student:
        _placeholder_figure(
            "M8 recognition",
            "No saved crops were found under outputs/cells/ yet.",
            config.FIGURES / "m8_score_distributions.png",
        )
        _placeholder_figure(
            "M8 recognition",
            "No saved crops were found under outputs/cells/ yet.",
            config.FIGURES / "m8_far_frr_curve.png",
        )
        _placeholder_figure(
            "M8 recognition",
            "No saved crops were found under outputs/cells/ yet.",
            config.FIGURES / "m8_feature_comparison.png",
        )
        _placeholder_figure(
            "M8 recognition",
            "No saved crops were found under outputs/cells/ yet.",
            config.FIGURES / "m8_similarity_matrix.png",
        )
        _placeholder_figure(
            "M8 recognition",
            "No saved crops were found under outputs/cells/ yet.",
            config.FIGURES / "m8_orb_matches.png",
        )
        _placeholder_figure(
            "M8 recognition",
            "No saved crops were found under outputs/cells/ yet.",
            config.FIGURES / "m8_normalisation_steps.png",
        )
        _placeholder_figure(
            "M8 recognition",
            "No saved crops were found under outputs/cells/ yet.",
            config.FIGURES / "m8_flagged_example.png",
        )
        empty = np.array([], dtype=np.float64)
        return EvaluationSummary(
            genuine=[],
            impostor=[],
            thresholds=empty,
            fars=empty,
            frrs=empty,
            eer_threshold=float("nan"),
            eer_rate=float("nan"),
            separation={},
            best_feature="",
        )

    genuine_scores = [pair.score.combined for pair in genuine]
    impostor_scores = [pair.score.combined for pair in impostor]

    matcher.save_pair_distribution_figure(genuine_scores, impostor_scores, config.FIGURES / "m8_score_distributions.png")
    thresholds, fars, frrs, eer_threshold, eer_rate = matcher.threshold_curve(genuine_scores, impostor_scores)
    matcher.save_threshold_curve_figure(thresholds, fars, frrs, eer_threshold, eer_rate, config.FIGURES / "m8_far_frr_curve.png")

    separation = matcher.feature_separation(genuine, impostor)
    matcher.save_feature_separation_figure(separation, config.FIGURES / "m8_feature_comparison.png")
    best_feature = max(separation, key=separation.get) if separation else ""

    config.MATCH_THRESHOLD = eer_threshold

    representative_student = max(samples_by_student.items(), key=lambda item: len(item[1]))[1]
    if len(representative_student) >= 2:
        matcher.save_similarity_matrix_figure(
            representative_student,
            config.FIGURES / "m8_similarity_matrix.png",
            title=f"Combined similarity matrix for {representative_student[0].student_index}",
        )
        save_flagged_example(representative_student, config.FIGURES / "m8_flagged_example.png")
    else:
        _placeholder_figure(
            "Similarity matrix",
            "Need at least two samples from one student to build the heatmap.",
            config.FIGURES / "m8_similarity_matrix.png",
        )
        _placeholder_figure(
            "Flagged example",
            "Need at least two samples from one student to flag an outlier.",
            config.FIGURES / "m8_flagged_example.png",
        )

    best_orb_pair = max((pair for pair in genuine if pair.score.orb > 0.0), key=lambda pair: pair.score.orb, default=None)
    if best_orb_pair is not None:
        save_orb_matches(best_orb_pair.left, best_orb_pair.right, config.FIGURES / "m8_orb_matches.png")
    else:
        _placeholder_figure(
            "ORB matches",
            "No genuine pair produced confident ORB matches.",
            config.FIGURES / "m8_orb_matches.png",
        )

    first_sample = next(iter(samples_by_student.values()))[0]
    save_normalisation_steps(first_sample, config.FIGURES / "m8_normalisation_steps.png")

    if not save_only:
        plt.show()

    return EvaluationSummary(
        genuine=genuine,
        impostor=impostor,
        thresholds=thresholds,
        fars=fars,
        frrs=frrs,
        eer_threshold=eer_threshold,
        eer_rate=eer_rate,
        separation=separation,
        best_feature=best_feature,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the M8 signature-recognition figures.")
    parser.add_argument("--show", action="store_true", help="open the generated figures after saving them")
    args = parser.parse_args()

    summary = build_evaluation(save_only=not args.show)
    ground_truth = load_ground_truth_summary()

    if summary.genuine:
        print(f"Ground truth: {ground_truth.genuine_pair_count} genuine pairs across {len(ground_truth.present_counts)} students")
        print(f"Measured EER threshold: {summary.eer_threshold:.3f} (EER {summary.eer_rate:.3f})")
        if summary.best_feature:
            print(f"Best separating feature: {summary.best_feature}")
        print("Figures saved to outputs/figures/")
    else:
        print("No crops available yet under outputs/cells/; placeholder figures were written instead.")


if __name__ == "__main__":
    main()
