"""The five signature comparison features.

Each feature catches something different about a signature, and none of them
is enough alone:

* SSIM sees overall shape agreement, but is thrown off by a small shift or
  rotation that ``preprocess_sig.normalise_signature`` doesn't fully absorb.
* HOG catches the direction strokes travel in, independent of exact position.
* Hu moments describe shape while ignoring size and rotation — good at
  catching "this is a completely different loop pattern", weak at catching
  fine stroke detail.
* ORB catches loops and crossings — where pen paths cross themselves — which
  none of the others look at directly.
* ``shape_stats`` is this project's own hand-made feature set: density,
  aspect, stroke length, how many enclosed loops, and where the ink sits.

Every ``*_similarity`` function in this module returns a score in ``[0, 1]``
where ``1`` means identical. ``matcher.compare`` combines the five into one
weighted score — see ``config.SCORE_WEIGHTS``.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.feature import hog
from skimage.measure import euler_number
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize

from src import config


def _clip01(score: float) -> float:
    """Force a score into ``[0, 1]``, this module's shared convention.

    Most of the five features land in range by construction (a cosine
    similarity of non-negative vectors, a ``1 / (1 + distance)`` conversion).
    This is the explicit safety net for the ones that don't: notably
    :func:`orb_similarity`, where ``knnMatch`` is not one-to-one and a good
    match count can — in a real, observed case, not just in theory — exceed
    the smaller keypoint count it's divided by. Every ``*_score`` /
    ``*_similarity`` function in this module routes its return value through
    here, so nothing downstream in ``matcher.compare`` has to defend against
    an out-of-range float on top of everything else it does.
    """
    return float(np.clip(score, 0.0, 1.0))


def ssim_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity between two normalised signature masks.

    Parameters
    ----------
    a, b:
        Two masks of identical shape, as produced by
        ``preprocess_sig.normalise_signature``.

    Returns
    -------
    A score in ``[0, 1]``. ``structural_similarity`` itself returns a value
    in ``[-1, 1]`` (negative only for strongly anti-correlated images, which
    two ink masks essentially never are) — clipped here so every feature in
    this module shares the same range, which is what lets
    ``matcher.compare`` combine them with a plain weighted sum.
    """
    if a.shape != b.shape:
        raise ValueError(f"ssim_similarity needs equal shapes, got {a.shape} and {b.shape}")

    score, _ = structural_similarity(a, b, full=True)
    return _clip01(score)


def hog_vector(norm: np.ndarray) -> np.ndarray:
    """Histogram-of-oriented-gradients descriptor of a normalised signature.

    HOG summarises which direction the ink travels in, cell by cell — it is
    sensitive to stroke *direction* in a way pixel-level SSIM is not, which
    is what makes it catch a signature whose overall silhouette is similar
    but whose strokes actually run a different way (a mismatch SSIM alone
    tends to miss, since it only looks at where pixels agree).

    Parameters use ``config.HOG_ORIENTATIONS``, ``config.HOG_PPC`` and
    ``config.HOG_CPB``, so the descriptor length only changes if those
    change — kept centralised for the same reason every other tunable is.
    """
    vector = hog(
        norm,
        orientations=config.HOG_ORIENTATIONS,
        pixels_per_cell=config.HOG_PPC,
        cells_per_block=config.HOG_CPB,
        feature_vector=True,
    )
    return vector.astype(np.float64)


def hog_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two signatures' HOG descriptors.

    HOG vectors are non-negative (they are histograms), so their cosine
    similarity already falls in ``[0, 1]`` — no conversion needed, unlike a
    signed distance. A zero-norm vector (a totally blank normalised image,
    e.g. comparing against an empty cell) has no defined direction, so that
    case returns ``0.0`` rather than dividing by zero.
    """
    va, vb = hog_vector(a), hog_vector(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    cosine = float(np.dot(va, vb) / denom)
    return _clip01(cosine)


def hu_moments(norm: np.ndarray) -> np.ndarray:
    """Log-scaled Hu moments of a normalised signature.

    Hu moments describe shape in a way that is invariant to translation,
    scale and rotation — useful here specifically because it catches gross
    shape differences (a completely different loop layout) even if
    ``normalise_signature`` left a slight scale or rotation mismatch that
    would otherwise confuse a pixel-level feature.

    Raw Hu moments span many orders of magnitude (roughly ``1e-1`` to
    ``1e-15``), which would let the smallest moment be swamped by
    floating-point noise in any plain distance. The standard fix — used
    here — is a sign-preserving log: ``-sign(h) * log10(|h|)``. A moment
    that is exactly zero stays zero rather than producing ``-inf``.
    """
    moments = cv2.moments(norm)
    hu = cv2.HuMoments(moments).flatten()

    log_hu = np.zeros_like(hu)
    nonzero = hu != 0
    log_hu[nonzero] = -np.sign(hu[nonzero]) * np.log10(np.abs(hu[nonzero]))
    return log_hu


def hu_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity between two signatures' log-scaled Hu moment vectors.

    The raw comparison is a distance (0 = identical shape, growing with
    difference), so it is converted to a ``[0, 1]`` similarity with
    ``1 / (1 + distance)`` — the standard distance-to-similarity conversion
    used throughout this module, chosen because it maps 0 to exactly 1.0
    and decays smoothly rather than needing a hand-picked cap.
    """
    distance = float(np.linalg.norm(hu_moments(a) - hu_moments(b)))
    return _clip01(1.0 / (1.0 + distance))


def orb_keypoints(norm: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """ORB keypoints and descriptors of a normalised signature.

    ORB catches something the other four features do not: loops and
    crossings — the specific points where a pen path crosses itself or
    curls back. Two signatures that look similar in overall shape but differ
    in exactly where their loops are (a common way a forger's copy fails)
    show up here even when SSIM and Hu moments are fooled.

    Returns ``(keypoints, descriptors)``. ``descriptors`` is ``None`` when
    ORB finds nothing to describe — an almost blank normalised image has no
    corners for it to detect, and that is a valid outcome, not an error.
    """
    orb = cv2.ORB_create(nfeatures=config.ORB_N_FEATURES)
    keypoints, descriptors = orb.detectAndCompute(norm, None)
    return keypoints, descriptors


def orb_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of ORB keypoints that find a confident match, via Lowe's ratio test.

    Every keypoint in ``a`` is matched against its two nearest neighbours in
    ``b``. A match only counts as "good" when the best neighbour is
    meaningfully closer than the second-best (``config.ORB_LOWE_RATIO``) —
    that is Lowe's ratio test, and it exists because with descriptors this
    short, plenty of points have an equally-good runner-up purely by chance;
    keeping only matches where one candidate clearly wins is what makes the
    count mean something.

    The result is good matches divided by the smaller of the two keypoint
    counts, so a signature that is a strict subset of another's strokes (or
    vice versa) doesn't get penalised just for having fewer points to offer.
    Returns ``0.0`` whenever either signature has no descriptors at all —
    there is nothing to match, which is a real answer, not missing data.
    """
    _, desc_a = orb_keypoints(a)
    _, desc_b = orb_keypoints(b)

    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_a, desc_b, k=2)

    good = 0
    for pair in pairs:
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < config.ORB_LOWE_RATIO * second.distance:
            good += 1

    smaller_side = min(len(desc_a), len(desc_b))
    if not smaller_side:
        return 0.0
    # `knnMatch` is not one-to-one — several `desc_a` points can each pick
    # the same `desc_b` neighbour — so `good` is not actually bounded by
    # `smaller_side` and this ratio can exceed 1.0 in practice, not just in
    # theory. `_clip01` is load-bearing here, not decorative.
    return _clip01(good / smaller_side)


def shape_stats(norm: np.ndarray) -> dict:
    """This project's own hand-made feature set — no library does the
    combining here, only individual building blocks (moments, skeletonize).

    Measures six things about the ink itself, each catching something the
    library features don't look at directly:

    * ``ink_density`` — ink pixels over total pixels. A hurried scrawl and a
      careful signature can have the same silhouette but very different
      density.
    * ``aspect`` — width over height of the ink's own bounding box (not the
      canvas — the canvas is always the same shape after normalisation).
    * ``stroke_length`` — pixels in the skeletonised stroke: how far the pen
      actually travelled, independent of how thick the ink is.
    * ``euler_number`` — connected components minus enclosed loops. A
      signature with a looped ``l`` or a crossed ``t`` has a different Euler
      number to one without, regardless of overall shape.
    * ``centre_of_mass`` — where the ink sits within the normalised canvas,
      as an ``(x, y)`` fraction, so it's comparable however big the canvas is.
    * ``slant_angle`` — the dominant tilt of the ink, in radians, from the
      image's own second moments — the same idea handwriting analysts call
      slant.

    Also returns the raw ``h_profile`` / ``v_profile`` projection profiles
    (ink pixels summed per row / per column) for :func:`custom_similarity`
    to correlate directly, since a profile is a shape in its own right and
    reducing it to a single number here would throw that away.
    """
    h, w = norm.shape[:2]
    ink = norm > 0
    ink_count = int(ink.sum())

    ink_density = ink_count / (h * w) if h * w else 0.0

    if ink_count > 0:
        ys, xs = np.nonzero(ink)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
        aspect = bbox_w / bbox_h if bbox_h else 0.0
        centre_of_mass = (float(xs.mean()) / w, float(ys.mean()) / h)
    else:
        aspect = 0.0
        centre_of_mass = (0.5, 0.5)

    stroke_length = int(skeletonize(ink).sum())
    euler = int(euler_number(ink, connectivity=2))

    moments = cv2.moments(norm)
    mu11, mu20, mu02 = moments["mu11"], moments["mu20"], moments["mu02"]
    slant_angle = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02) if ink_count > 0 else 0.0

    return {
        "ink_density": ink_density,
        "aspect": aspect,
        "stroke_length": stroke_length,
        "euler_number": euler,
        "centre_of_mass": centre_of_mass,
        "slant_angle": slant_angle,
        "h_profile": ink.sum(axis=1).astype(np.float64),
        "v_profile": ink.sum(axis=0).astype(np.float64),
    }


def _scalar_stats_similarity(stats_a: dict, stats_b: dict) -> float:
    """Similarity from the six scalar/tuple entries of :func:`shape_stats`.

    Each sub-score uses whichever conversion suits its own units — a plain
    ``1/(1+distance)`` for unbounded values, a relative difference for
    values with a natural scale, and wrap-around handling for the slant
    angle (91 degrees and -89 degrees are one degree apart, not 180) —
    then all six are averaged with equal weight.
    """
    a_density, b_density = stats_a["ink_density"], stats_b["ink_density"]
    density_sim = 1.0 - min(abs(a_density - b_density) / max(a_density, b_density, 1e-6), 1.0)

    aspect_sim = 1.0 / (1.0 + abs(stats_a["aspect"] - stats_b["aspect"]))

    a_len, b_len = stats_a["stroke_length"], stats_b["stroke_length"]
    stroke_sim = 1.0 - min(abs(a_len - b_len) / max(a_len, b_len, 1), 1.0)

    euler_sim = 1.0 / (1.0 + abs(stats_a["euler_number"] - stats_b["euler_number"]))

    com_ax, com_ay = stats_a["centre_of_mass"]
    com_bx, com_by = stats_b["centre_of_mass"]
    com_distance = math.hypot(com_ax - com_bx, com_ay - com_by)
    com_sim = 1.0 / (1.0 + com_distance * 10.0)  # fractions of the canvas are small; scale up

    angle_diff = abs(stats_a["slant_angle"] - stats_b["slant_angle"])
    angle_diff = min(angle_diff, math.pi - angle_diff)
    slant_sim = 1.0 - min(angle_diff / (math.pi / 2), 1.0)

    return _clip01(np.mean([density_sim, aspect_sim, stroke_sim, euler_sim, com_sim, slant_sim]))


def _profile_correlation(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """Pearson correlation between two projection profiles, mapped to ``[0, 1]``.

    A projection profile (ink pixels summed per row, or per column) is a
    shape in its own right — two signatures with the same scalar stats can
    still distribute their ink very differently along one axis (e.g. a flat
    scrawl versus one with a tall looping capital), and correlation is
    sensitive to exactly that kind of shape agreement in a way none of
    :func:`shape_stats`'s six scalar numbers are.

    Correlation is undefined when a profile has zero variance (a blank
    normalised image, or — in principle — a perfectly flat one). Two blank
    profiles are trivially identical, so that pair returns ``1.0``; a blank
    against a non-blank profile has nothing in common to correlate, so that
    returns ``0.0`` rather than raising on the division by zero.
    """
    std_a, std_b = profile_a.std(), profile_b.std()
    if std_a == 0.0 and std_b == 0.0:
        return 1.0
    if std_a == 0.0 or std_b == 0.0:
        return 0.0

    correlation = float(np.corrcoef(profile_a, profile_b)[0, 1])
    if np.isnan(correlation):
        return 0.0
    # Pearson correlation is [-1, 1]; map onto this module's [0, 1] convention.
    return (correlation + 1.0) / 2.0


def custom_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """This project's own similarity score — the ``shape_stats`` feature.

    Combines two things in equal measure:

    * the six scalar/tuple entries of :func:`shape_stats` (density, aspect,
      stroke length, Euler number, centre of mass, slant), via
      :func:`_scalar_stats_similarity`
    * how well the horizontal and vertical ink projection profiles line up,
      via :func:`_profile_correlation` on each axis, themselves averaged

    Equal weighting between "single numbers about the ink" and "where the
    ink actually sits along each axis" is a design choice, not a measured
    one — T5's feature-separation report is what actually justifies (or
    revises) it, the same way it justifies ``config.SCORE_WEIGHTS`` overall.
    """
    stats_a, stats_b = shape_stats(a), shape_stats(b)

    scalar_sim = _scalar_stats_similarity(stats_a, stats_b)

    h_corr = _profile_correlation(stats_a["h_profile"], stats_b["h_profile"])
    v_corr = _profile_correlation(stats_a["v_profile"], stats_b["v_profile"])
    profile_sim = (h_corr + v_corr) / 2.0

    return _clip01(0.5 * scalar_sim + 0.5 * profile_sim)