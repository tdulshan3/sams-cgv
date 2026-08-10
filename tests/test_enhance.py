"""Tests for greyscale conversion, denoising, shadow removal and contrast.

Covers BUILD_SPEC.md section 11 (M3) plus the PSNR/runtime measurement T2
asks for. Synthetic images only, small and fast, per section 11's note that
a test needing a full sheet photo is slow and tells you less.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio

from src.preprocess.enhance import denoise, enhance_contrast, remove_shadow, to_grey

DENOISE_METHODS = ["gaussian", "median", "bilateral", "nlmeans"]


def test_to_grey_shape_and_dtype():
    """to_grey output is 2-D uint8, same height and width as the input."""
    bgr = np.zeros((30, 50, 3), dtype=np.uint8)
    grey = to_grey(bgr, "luminosity")
    assert grey.ndim == 2
    assert grey.dtype == np.uint8
    assert grey.shape == (30, 50)


def test_luminosity_pure_red_pixel():
    """Luminosity of a pure red pixel (0, 0, 255) BGR is approximately 76."""
    red = np.array([[[0, 0, 255]]], dtype=np.uint8)
    grey = to_grey(red, "luminosity")
    assert abs(int(grey[0, 0]) - 76) <= 1


def test_denoise_reduces_variance_on_noisy_flat_patch():
    """denoise reduces variance on a synthetic noisy flat patch."""
    rng = np.random.default_rng(1)
    flat = np.full((60, 60), 180, dtype=np.uint8)
    noisy = np.clip(
        flat.astype(np.int16) + rng.normal(0, 20, flat.shape), 0, 255
    ).astype(np.uint8)

    for method in DENOISE_METHODS:
        result = denoise(noisy, method)
        assert result.var() < noisy.var()


def test_remove_shadow_flattens_linear_ramp():
    """remove_shadow on a linear brightness ramp reduces the left/right half
    difference that a shadow gradient would otherwise cause."""
    height, width = 200, 200
    ramp = np.tile(np.linspace(60, 220, width, dtype=np.uint8), (height, 1))

    def half_difference(image: np.ndarray) -> float:
        left = image[:, : width // 2].astype(np.float64).mean()
        right = image[:, width // 2 :].astype(np.float64).mean()
        return abs(left - right)

    flattened = remove_shadow(ramp)
    assert half_difference(flattened) < half_difference(ramp)


def test_unknown_method_raises_value_error():
    """An unknown method name raises ValueError for every method-selecting
    function, rather than silently falling back to a default."""
    grey = np.zeros((10, 10), dtype=np.uint8)
    bgr = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        to_grey(bgr, "not-a-method")
    with pytest.raises(ValueError):
        denoise(grey, "not-a-method")
    with pytest.raises(ValueError):
        enhance_contrast(grey, "not-a-method")


def test_denoise_psnr_and_runtime_metrics():
    """Every denoise method runs, improves PSNR over the noisy input, and
    reports a runtime: the comparison T2 asks for rather than a claim."""
    rng = np.random.default_rng(0)
    clean = np.full((80, 80), 200, dtype=np.uint8)
    noisy = np.clip(
        clean.astype(np.int16) + rng.normal(0, 25, clean.shape), 0, 255
    ).astype(np.uint8)
    noisy_psnr = peak_signal_noise_ratio(clean, noisy)

    for method in DENOISE_METHODS:
        start = time.perf_counter()
        result = denoise(noisy, method)
        elapsed = time.perf_counter() - start

        assert result.shape == noisy.shape
        assert elapsed >= 0.0
        result_psnr = peak_signal_noise_ratio(clean, result)
        assert result_psnr >= noisy_psnr
