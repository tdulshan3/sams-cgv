# M8 Contribution Notes

## What I built

M8 is the signature-recognition layer that compares crops from `outputs/cells/<sheet_date>/<index>.png` and decides whether a student's signature is consistent across sheets. The implementation lives in `src/recognise/preprocess_sig.py`, `src/recognise/features.py`, `src/recognise/matcher.py`, and `tools/eval_recognition.py`, with the tunable recognition settings in the M8 block of `src/config.py`.

The pipeline is split into three stages:

1. `preprocess_sig.py` normalises a raw ink mask into a fixed comparison box by trimming the ink, padding the crop, resizing it without distortion, and centring it by mass.
2. `features.py` extracts five complementary similarity signals: SSIM, HOG, Hu moments, ORB matches, and a custom hand-built similarity score.
3. `matcher.py` combines those features into one weighted score, prints pairwise tables for `investigate.py`, detects the least similar sheet for one student, and saves the per-student figures.

`tools/eval_recognition.py` builds the genuine and impostor pair sets, sweeps the combined-score threshold, measures FAR/FRR and EER, reports the best-separating feature, and writes the recognition figures under `outputs/figures/`.

## What the code does

The matcher now supports the full `investigate(index: str, save_only: bool = False) -> None` contract. For a given student it:

- loads every saved sample for that index directly from `outputs/cells/`
- prints a pairwise table with SSIM, HOG, HU, ORB, OWN, COMBINED, and a verdict per pair
- prints a final verdict that either confirms all samples match or names the sheet date that is least consistent with the rest
- saves `outputs/figures/m8_investigate_<index>.png`

For the outlier work, the matcher also builds a full similarity matrix for one student's samples and computes the mean similarity of each sample to the others. The sample with the lowest mean is treated as the outlier candidate and is highlighted in the saved figure.

For T5, the evaluator script compares all same-student pairs as genuine pairs and all cross-student pairs as impostor pairs. It then:

- plots the genuine and impostor score distributions on one histogram
- sweeps the combined threshold and plots FAR/FRR with the EER point marked
- measures which individual feature separates genuine and impostor pairs best
- writes the recognition figures at 150 dpi or higher

The real dataset ground truth still matters here: the CSV in `data/ground_truth.csv` gives six students with 41 genuine pairs total, and the code keeps that number honest instead of implying more data than exists.

## Figures and validation

The M8 figures are:

- `outputs/figures/m8_normalisation_steps.png`
- `outputs/figures/m8_score_distributions.png`
- `outputs/figures/m8_far_frr_curve.png`
- `outputs/figures/m8_similarity_matrix.png`
- `outputs/figures/m8_orb_matches.png`
- `outputs/figures/m8_feature_comparison.png`
- `outputs/figures/m8_flagged_example.png`
- `outputs/figures/m8_investigate_<index>.png` for the CLI investigation path

In this workspace there are no real crops under `outputs/cells/` yet, so the evaluator currently writes placeholder figures and prints a clear message instead of crashing. That keeps the code runnable before M6 and M7 land, while still exercising the full file and figure pipeline.

## Test coverage

`tests/test_recognition.py` covers the invariants the recognition stack depends on:

- self-similarity is effectively 1.0
- a blank image scores low against a real signature
- a 10px shift still scores highly
- a 1.5x scale still scores highly
- `normalise_signature()` returns exactly `config.SIG_NORM_SIZE`
- a single-sample `investigate()` call prints a clear no-exception message

That gives the module a tight behavioral fence around the comparison path without relying on the missing real crops.

## Notes for the next step

When real crops are present under `outputs/cells/`, rerun `tools/eval_recognition.py` to refresh the EER measurement and the feature-separation chart against the real data rather than the placeholder branch state in this workspace.
# M8 — Signature Recognition

Individual contribution notes. Two pages minimum, per the coursework brief:
what you built, the techniques you used and why, the problems you hit, and
the figures in `outputs/` that back it up.

## What I built

_TBD_

## Techniques and libraries

_TBD_

## Problems and how I solved them

_TBD_

## Evidence

_TBD_
