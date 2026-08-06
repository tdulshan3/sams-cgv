"""Signature normalisation.

Two signatures cannot be compared until they sit in the same box at the same
scale. A raw signature crop varies in position (where the pen started),
size (how big the student signed), and background (whatever the cell crop
picked up). This module removes all three variables before any comparison
happens in ``matcher.py``.

Pipeline, in order: trim to the ink bounding box -> pad back to the target
aspect ratio -> centre by centre of mass -> resize to a fixed box.
"""

from __future__ import annotations

import cv2
import numpy as np

from src import config


def trim_to_ink(mask: np.ndarray, pad: int = 0) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop ``mask`` down to the bounding box of its non-zero (ink) pixels.

    Parameters
    ----------
    mask:
        Single channel image, ink = 255, background = 0.
    pad:
        Extra pixels of background kept on every side of the ink bounding
        box, clamped to the image edges.

    Returns
    -------
    A tuple of the trimmed mask and the bounding box used, as
    ``(x, y, w, h)`` in the input image's coordinates. If ``mask`` has no
    ink at all, the mask is returned unchanged and the box covers the
    whole image — there is nothing to trim to, and callers should treat
    an all-blank input as a "no signature here" signal rather than a crash.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        h, w = mask.shape[:2]
        return mask, (0, 0, w, h)

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(mask.shape[1] - 1, x2 + pad)
    y2 = min(mask.shape[0] - 1, y2 + pad)

    trimmed = mask[y1 : y2 + 1, x1 : x2 + 1]
    return trimmed, (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def _centre_of_mass(mask: np.ndarray) -> tuple[float, float]:
    """Return ``(row, col)`` centre of mass of the non-zero pixels.

    Falls back to the geometric centre of the image when the mask is blank,
    so callers never have to special-case a zero-ink signature themselves.
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        h, w = mask.shape[:2]
        return h / 2.0, w / 2.0
    return float(ys.mean()), float(xs.mean())


def centre_by_mass(mask: np.ndarray, canvas_size: tuple[int, int]) -> np.ndarray:
    """Place ``mask`` on a blank canvas so its centre of mass sits in the middle.

    Parameters
    ----------
    mask:
        A trimmed ink mask (see :func:`trim_to_ink`), smaller than or equal
        to ``canvas_size`` in both dimensions.
    canvas_size:
        ``(width, height)`` of the output canvas. ``mask`` is placed inside
        it, cropped further only if it does not fit even after centring.

    Two identical signatures placed at different starting positions inside
    their cell would otherwise score as dissimilar on a pixel-level
    comparison such as SSIM. Centring by centre of mass — rather than by
    bounding-box centre — is more stable for signatures, since a long trailing
    flourish can skew a bounding-box centre far from where most of the ink
    actually sits.
    """
    canvas_w, canvas_h = canvas_size
    canvas = np.zeros((canvas_h, canvas_w), dtype=mask.dtype)

    mh, mw = mask.shape[:2]
    com_y, com_x = _centre_of_mass(mask)

    # Where the centre of mass should land on the canvas.
    target_y, target_x = canvas_h / 2.0, canvas_w / 2.0

    # Top-left corner to paste `mask` at so its centre of mass lands there.
    paste_x = int(round(target_x - com_x))
    paste_y = int(round(target_y - com_y))

    # Source region of `mask` to copy (handles the case where the shifted
    # placement would run off either edge of the canvas).
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = min(mw, canvas_w - paste_x)
    src_y2 = min(mh, canvas_h - paste_y)

    dst_x1 = max(0, paste_x)
    dst_y1 = max(0, paste_y)
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]

    return canvas


def resize_keep_aspect(mask: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Scale ``mask`` so it fits inside ``target_size`` without distorting it.

    Parameters
    ----------
    mask:
        The image to scale.
    target_size:
        ``(width, height)`` of the box the result must fit inside.

    A student who signs small and a student who signs large must not be
    told apart by size alone once normalised — but stretching width and
    height independently would also warp the *shape* of every stroke, which
    is exactly what the shape-based features are supposed to measure. So the
    scale factor is the same in both directions, chosen so the longer side
    of the ink just fits the corresponding side of the box.
    """
    target_w, target_h = target_size
    h, w = mask.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target_h, target_w), dtype=mask.dtype)

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(mask, (new_w, new_h), interpolation=interp)


def normalise_signature(mask: np.ndarray, size: tuple[int, int] = config.SIG_NORM_SIZE) -> np.ndarray:
    """Bring a raw signature mask into a fixed box for comparison.

    Trims to the ink bounding box, scales the ink to fit ``size`` while
    keeping its aspect ratio, then centres it on a canvas of exactly
    ``size`` by centre of mass. The result always has shape
    ``(size[1], size[0])`` regardless of the input's shape — that guarantee
    is what lets ``matcher.compare`` treat every pair of signatures as
    directly comparable arrays.

    Parameters
    ----------
    mask:
        Ink mask, ink = 255, background = 0. A colour crop must be turned
        into a mask (M6's ``ink_mask`` / M4's ``clean_signature_crop``)
        before it reaches this function — normalisation only handles shape,
        not colour.
    size:
        ``(width, height)`` of the output box. Defaults to
        ``config.SIG_NORM_SIZE``.

    A blank input (no ink at all) is not an error: it normalises to a blank
    canvas of the right size, and it is ``matcher.compare``'s job to decide
    what a blank-versus-signed comparison means.
    """
    target_w, target_h = size
    trimmed, _ = trim_to_ink(mask, pad=config.SIG_NORM_PAD)
    scaled = resize_keep_aspect(trimmed, size)
    canvas = centre_by_mass(scaled, (target_w, target_h))
    return canvas