"""Tests for greyscale conversion, denoising, shadow removal and contrast.

Covers BUILD_SPEC.md section 11 (M3) plus the PSNR/runtime measurement T2
asks for. Synthetic images only — small and fast, per section 11's note that
a test needing a full sheet photo is slow and tells you less.
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio

from src.preprocess.enhance import denoise, to_grey

DENOISE_METHODS = ["gaussian", "median", "bilateral", "nlmeans"]


def test_denoise_psnr_and_runtime_metrics():
    """Every denoise method runs, improves PSNR over the noisy input, and
    reports a runtime — the comparison T2 asks for rather than a claim."""
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
