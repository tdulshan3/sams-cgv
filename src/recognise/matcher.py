"""Compare one student's signatures across sheets and report a mismatch.

This is the module ``investigate.py`` calls into — see that file's docstring
for the command line contract. Everything here assumes ``investigate.py`` (via
``src.cli.signature_samples``) has already confirmed there are at least two
saved signatures for the student; this module does not repeat that check on
the CLI path, but every function here is still safe to call directly (e.g.
from tests) with too few samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.recognise import features
from src.recognise.preprocess_sig import normalise_signature
from src.utils.logging import get_logger

log = get_logger("recognise.matcher")

SHEET_DATE_FORMAT = "%d.%m.%Y"
"""Sheet folder names are e.g. ``12.07.2019`` — day first, so a plain string
sort puts July before May. Every place that needs chronological order must
parse through this format rather than sorting the folder name directly.
"""


@dataclass
class SignatureSample:
    """One student's signature, as saved by M6 for a single sheet.

    ``mask`` is what every comparison in this module actually operates on —
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
    tag) for any folder that doesn't parse — a stray or renamed folder should
    not crash a comparison run, just sort last and predictably.
    """
    try:
        return (0, datetime.strptime(cell_dir.name, SHEET_DATE_FORMAT).isoformat())
    except ValueError:
        log.warning("cell folder %s is not a %s date, sorting it last", cell_dir.name, SHEET_DATE_FORMAT)
        return (1, cell_dir.name)


def _read_mask(cell_dir: Path, index: str) -> np.ndarray | None:
    """Load the saved ink mask for one student in one sheet's cell folder.

    Returns ``None`` rather than raising when the mask file is absent — this
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

    Plain saturation-or-darkness threshold — deliberately not a real ink
    segmentation. This exists so ``investigate.py`` can be demonstrated end to
    end before M6's module merges; the moment a real ``<index>_mask.png`` is
    on disk, :func:`load_samples` prefers it and this function is never
    called for that sample. Not a substitute for M6's work and not meant to
    be tuned — it is a placeholder, in the same spirit as ``src/stubs.py``.
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
    exception — an empty or short list is an ordinary state of the project
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
                "no saved mask for %s on %s yet — using a rough bootstrap mask",
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
    identical — the shared convention every function in ``features.py``
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
    """One of ``"match"``, ``"mismatch"``, ``"uncertain"`` — see :func:`_verdict`."""


def _weighted_combine(ssim: float, hog: float, hu: float, orb: float, custom: float) -> float:
    """Combine the five feature scores into one, using ``config.SCORE_WEIGHTS``.

    A plain weighted sum, not an average of some cleverer kind — every
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
    "uncertain" — rather than forcing a confident-looking match or mismatch
    out of a borderline number — is, per the brief, a strength of this
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
        Ink masks, ink = 255, background = 0 — *not* yet normalised.
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
"""One signature cannot be compared with anything — mirrors investigate.py's
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
        ``outputs/figures/`` and print no window — for headless / CI runs.

    A student with fewer than :data:`MIN_SAMPLES` saved signatures is not an
    error: there is nothing to compare yet, so this prints a plain
    explanation and returns rather than raising.
    """
    samples = load_samples(index)
    if len(samples) < MIN_SAMPLES:
        print(
            f"student {index} has {len(samples)} saved signature "
            f"{'sample' if len(samples) == 1 else 'samples'} — "
            f"at least {MIN_SAMPLES} are needed to compare"
        )
        return

    # T3/T4: per-pair feature scoring and verdicts.
    # T6: outlier detection across the whole set.
    # T7: table + verdict printout, figure saving (uses `save_only`).
    raise NotImplementedError(
        "comparison logic lands in T3-T7 — sample loading (T1) is done and tested"
    )