from __future__ import annotations

import cv2
import numpy as np
import pytest

from src import config
from src.recognise.matcher import SignatureSample, compare, investigate
from src.recognise.preprocess_sig import normalise_signature


def make_signature_mask(size: tuple[int, int] = (150, 150)) -> np.ndarray:
    """Draw a simple synthetic signature-like stroke on a blank canvas."""
    canvas = np.zeros(size, dtype=np.uint8)
    points = np.array([[20, 100], [35, 40], [55, 95], [75, 55], [95, 85], [120, 35]], np.int32)
    cv2.polylines(canvas, [points], False, 255, thickness=8, lineType=cv2.LINE_AA)
    return canvas


def embed_mask(mask: np.ndarray, canvas_size: tuple[int, int] = (400, 400), top_left: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Place ``mask`` into a larger blank canvas at ``top_left``."""
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    row, col = top_left
    canvas[row : row + mask.shape[0], col : col + mask.shape[1]] = mask
    return canvas


def centre_scaled_mask(mask: np.ndarray, scale: float, canvas_size: tuple[int, int] = (400, 400)) -> np.ndarray:
    """Scale ``mask`` and centre it on a blank canvas of the same size."""
    scaled = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    row = (canvas.shape[0] - scaled.shape[0]) // 2
    col = (canvas.shape[1] - scaled.shape[1]) // 2
    canvas[row : row + scaled.shape[0], col : col + scaled.shape[1]] = scaled
    return canvas


def test_compare_self_similarity_is_close_to_one() -> None:
    mask = make_signature_mask()
    score = compare(mask, mask)
    assert score.combined == pytest.approx(1.0, abs=1e-6)


def test_compare_against_blank_is_low() -> None:
    mask = make_signature_mask()
    blank = np.zeros_like(mask)
    score = compare(mask, blank)
    assert score.combined < 0.35


def test_compare_shifted_signature_still_scores_high() -> None:
    mask = make_signature_mask()
    shifted = embed_mask(mask, top_left=(10, 10))
    original = embed_mask(mask, top_left=(30, 30))
    score = compare(original, shifted)
    assert score.combined > 0.9


def test_compare_scaled_signature_still_scores_high() -> None:
    base = make_signature_mask()
    original = embed_mask(base)
    scaled = centre_scaled_mask(base, 1.5)
    score = compare(original, scaled)
    assert score.combined > 0.7


def test_normalise_signature_returns_config_size() -> None:
    mask = make_signature_mask()
    normalised = normalise_signature(mask)
    assert normalised.shape == (config.SIG_NORM_SIZE[1], config.SIG_NORM_SIZE[0])


def test_investigate_single_sample_prints_clear_message(monkeypatch, capsys) -> None:
    mask = make_signature_mask()
    sample = SignatureSample(student_index="10000409", sheet_date="31.05.2019", mask=mask, crop=None)
    monkeypatch.setattr("src.recognise.matcher.load_samples", lambda index: [sample])

    investigate("10000409", save_only=True)

    output = capsys.readouterr().out
    assert "at least 2 are needed to compare" in output
