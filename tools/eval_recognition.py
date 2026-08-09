"""Measure the signature matcher, and set its threshold from the measurement.

M8 task T5. Five sheets means at most five samples per student and no known
forgeries, so a model cannot be trained and a threshold cannot be validated in
the usual way. The way round it:

* a student's signatures compared with each other  -> **genuine** pairs
* a student's compared with everyone else's        -> **impostor** pairs

Two score distributions, a cut-off where they separate, and real numbers for
False Accept Rate, False Reject Rate and the Equal Error Rate where the two
cross. That turns "we picked 0.62" into an experiment.

Usage::

    python tools/eval_recognition.py            # measure and report
    python tools/eval_recognition.py --apply    # also write the EER threshold
                                                # into src/config.py
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.recognise.matcher import compare  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("eval")

FEATURES = ("ssim", "hog", "hu", "orb", "custom", "combined")


def collect_samples() -> dict[str, list[tuple[str, Path]]]:
    """Every saved signature crop, grouped by student index.

    Reads ``outputs/cells/<sheet_date>/<index>.png`` directly rather than the
    database, so the experiment runs on whatever ``sams.py`` last produced.
    """
    samples: dict[str, list[tuple[str, Path]]] = {}
    if not config.CELLS.is_dir():
        return samples

    for folder in sorted(config.CELLS.iterdir()):
        if not folder.is_dir():
            continue
        # The matcher compares ink masks, not the colour crops — normalisation
        # and every feature downstream expect a binary stroke image.
        for mask in sorted(folder.glob("*_mask.png")):
            index = mask.stem.removesuffix("_mask")
            if not index.isdigit():
                continue
            samples.setdefault(index, []).append((folder.name, mask))
    return samples


def score_pairs(samples: dict[str, list[tuple[str, Path]]]) -> tuple[list[dict], list[dict]]:
    """Score every genuine pair and every impostor pair.

    Returns:
        ``(genuine, impostor)``, each a list of per-feature score dicts.
    """
    import cv2

    def load(path: Path) -> np.ndarray:
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    genuine: list[dict] = []
    for index, entries in samples.items():
        for (date_a, a), (date_b, b) in itertools.combinations(entries, 2):
            score = compare(load(a), load(b))
            genuine.append({f: float(getattr(score, f)) for f in FEATURES})
        log.info("student %s: %d samples", index, len(entries))

    impostor: list[dict] = []
    indices = sorted(samples)
    for index_a, index_b in itertools.combinations(indices, 2):
        for date_a, a in samples[index_a]:
            for date_b, b in samples[index_b]:
                score = compare(load(a), load(b))
                impostor.append({f: float(getattr(score, f)) for f in FEATURES})

    return genuine, impostor


def far_frr(genuine: list[float], impostor: list[float], thresholds: np.ndarray):
    """False accept and false reject rate at each threshold.

    FAR is the share of impostor pairs scoring at or above the threshold — the
    forgeries we would wave through. FRR is the share of genuine pairs scoring
    below it — the real signatures we would wrongly flag.
    """
    genuine_scores = np.asarray(genuine)
    impostor_scores = np.asarray(impostor)
    far = np.array([(impostor_scores >= t).mean() for t in thresholds])
    frr = np.array([(genuine_scores < t).mean() for t in thresholds])
    return far, frr


def equal_error_rate(thresholds: np.ndarray, far: np.ndarray, frr: np.ndarray):
    """The threshold where FAR and FRR cross, and the rate there."""
    crossing = int(np.argmin(np.abs(far - frr)))
    return float(thresholds[crossing]), float((far[crossing] + frr[crossing]) / 2)


def separation(genuine: list[float], impostor: list[float]) -> float:
    """d-prime: how many pooled standard deviations apart the two means sit.

    A single number for "how well does this feature tell the two apart", so the
    features can be ranked against each other rather than eyeballed.
    """
    g, i = np.asarray(genuine), np.asarray(impostor)
    spread = np.sqrt((g.var() + i.var()) / 2)
    return float(abs(g.mean() - i.mean()) / spread) if spread > 0 else 0.0


def plot_distributions(genuine: list[float], impostor: list[float], threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=config.FIGURE_DPI)
    bins = np.linspace(0, 1, 26)
    axis.hist(genuine, bins=bins, alpha=0.65, label=f"genuine pairs (n={len(genuine)})", color="#2E7D32")
    axis.hist(impostor, bins=bins, alpha=0.65, label=f"impostor pairs (n={len(impostor)})", color="#C62828")
    axis.axvline(threshold, color="black", linestyle="--", linewidth=1.6,
                 label=f"threshold from EER = {threshold:.3f}")
    axis.set_xlabel("combined similarity score")
    axis.set_ylabel("pairs")
    axis.set_title("M8 — genuine against impostor signature scores")
    axis.legend(fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(config.FIGURES / "m8_score_distributions.png", dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote outputs/figures/m8_score_distributions.png")


def plot_far_frr(thresholds, far, frr, eer_threshold, eer) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=config.FIGURE_DPI)
    axis.plot(thresholds, far * 100, label="FAR — impostors accepted", color="#C62828")
    axis.plot(thresholds, frr * 100, label="FRR — genuine rejected", color="#2E7D32")
    axis.axvline(eer_threshold, color="black", linestyle="--", linewidth=1.4)
    axis.plot([eer_threshold], [eer * 100], "ko", markersize=7,
              label=f"EER = {eer:.1%} at {eer_threshold:.3f}")
    axis.set_xlabel("threshold")
    axis.set_ylabel("rate (%)")
    axis.set_title("M8 — error rates against threshold")
    axis.legend(fontsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(config.FIGURES / "m8_far_frr_curve.png", dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote outputs/figures/m8_far_frr_curve.png")


def plot_feature_comparison(rankings: list[tuple[str, float]]) -> None:
    names = [name for name, _ in rankings]
    values = [value for _, value in rankings]
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=config.FIGURE_DPI)
    axis.barh(names, values, color="#e8eef7", edgecolor="#2b4c7e")
    axis.invert_yaxis()
    axis.set_xlabel("separation (d-prime, higher is better)")
    axis.set_title("M8 — which feature tells genuine from impostor best")
    axis.spines[["top", "right"]].set_visible(False)
    for y, value in enumerate(values):
        axis.text(value + max(values) * 0.02, y, f"{value:.2f}", va="center", fontsize=9)
    axis.set_xlim(0, max(values) * 1.2 if values else 1)
    figure.tight_layout()
    figure.savefig(config.FIGURES / "m8_feature_comparison.png", dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    log.info("wrote outputs/figures/m8_feature_comparison.png")


def apply_threshold(value: float) -> None:
    """Write the measured threshold into ``src/config.py``."""
    path = config.ROOT / "src" / "config.py"
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"^MATCH_THRESHOLD = [0-9.]+",
        f"MATCH_THRESHOLD = {value:.3f}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count:
        path.write_text(updated, encoding="utf-8")
        log.info("MATCH_THRESHOLD set to %.3f in src/config.py", value)
    else:
        log.warning("could not find MATCH_THRESHOLD in src/config.py")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eval_recognition.py", description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the measured EER threshold into src/config.py")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    samples = {k: v for k, v in collect_samples().items() if len(v) >= 2}
    if len(samples) < 2:
        print("error: need at least two students with two crops each")
        print("       run: python sams.py data/sheets/<sheet>.png data/info.xml")
        return 2

    genuine, impostor = score_pairs(samples)
    print()
    print(f"students {len(samples)} | genuine pairs {len(genuine)} | impostor pairs {len(impostor)}")

    thresholds = np.linspace(0.0, 1.0, 501)
    g_combined = [s["combined"] for s in genuine]
    i_combined = [s["combined"] for s in impostor]
    far, frr = far_frr(g_combined, i_combined, thresholds)
    eer_threshold, eer = equal_error_rate(thresholds, far, frr)

    print()
    print(f"{'':<10}{'genuine mean':>14}{'impostor mean':>15}{'separation':>12}")
    print("-" * 51)
    rankings = []
    for feature in FEATURES:
        g = [s[feature] for s in genuine]
        i = [s[feature] for s in impostor]
        d = separation(g, i)
        rankings.append((feature, d))
        print(f"{feature:<10}{np.mean(g):>14.3f}{np.mean(i):>15.3f}{d:>12.2f}")

    print()
    print(f"Equal Error Rate : {eer:.1%} at threshold {eer_threshold:.3f}")
    print(f"  FAR at that point: {far[np.argmin(np.abs(far - frr))]:.1%}")
    print(f"  FRR at that point: {frr[np.argmin(np.abs(far - frr))]:.1%}")
    print(f"  currently configured MATCH_THRESHOLD = {config.MATCH_THRESHOLD}")

    plot_distributions(g_combined, i_combined, eer_threshold)
    plot_far_frr(thresholds, far, frr, eer_threshold, eer)
    plot_feature_comparison(sorted(rankings, key=lambda r: -r[1]))

    if args.apply:
        apply_threshold(eer_threshold)
    else:
        print()
        print("re-run with --apply to write this threshold into src/config.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
