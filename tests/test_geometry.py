import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.io.image_loader import load_image
from src.preprocess.deskew import (
    estimate_skew_angle,
    four_point_warp,
    _order_points,
    GeometryStage,
)
from src.models import SheetMeta


def test_load_image_exceptions():
    """1. load_image raises FileNotFoundError on bad path, ValueError on non-image file."""
    with pytest.raises(FileNotFoundError):
        load_image("does_not_exist_at_all.jpg")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("This is not an image.")
        bad_path = f.name
        
    try:
        with pytest.raises(ValueError, match="not a decodable image"):
            load_image(bad_path)
    finally:
        os.remove(bad_path)


def test_estimate_skew_angle():
    """2. Synthetic white rectangle on dark background, rotated 7 degrees, corrected to within 1 degree."""
    # Create a black image
    h, w = 500, 500
    img = np.zeros((h, w), dtype=np.uint8)
    
    # Create a white rectangle
    rect_w, rect_h = 300, 200
    cx, cy = w // 2, h // 2
    
    # Get rotation matrix for 7 degrees
    M = cv2.getRotationMatrix2D((cx, cy), 7.0, 1.0)
    
    # Points of the rectangle centered at cx, cy
    pts = np.array([
        [-rect_w//2, -rect_h//2],
        [rect_w//2, -rect_h//2],
        [rect_w//2, rect_h//2],
        [-rect_w//2, rect_h//2]
    ], dtype=np.float32)
    
    # Add center offset to points
    pts[:, 0] += cx
    pts[:, 1] += cy
    
    # Rotate points
    pts_rotated = cv2.transform(np.array([pts]), M)[0]
    
    # Fill polygon
    cv2.fillPoly(img, [pts_rotated.astype(int)], 255)
    
    angle = estimate_skew_angle(img)
    
    # Should detect around 7.0 or -7.0 depending on the angle convention.
    # Hough lines might pick up 7 degrees.
    assert abs(abs(angle) - 7.0) <= 1.0


def test_four_point_warp_size():
    """3. four_point_warp on known quad gives expected output size (longest-opposite-edge)."""
    # Quad coordinates:
    # top-left: 0, 0
    # top-right: 100, 0  (width_top = 100)
    # bottom-right: 120, 50 (height_right = sqrt(20^2 + 50^2) = 53.85, width_bottom = 120)
    # bottom-left: 0, 50 (height_left = 50)
    
    corners = np.array([
        [0, 0],
        [100, 0],
        [120, 50],
        [0, 50]
    ], dtype=np.float32)
    
    # Dummy BGR image large enough
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    
    warped = four_point_warp(bgr, corners)
    
    # expected max width = max(100, 120) = 120
    # expected max height = max(50, 53) = 53
    expected_width = 120
    expected_height = 53
    
    assert warped.shape[:2] == (expected_height, expected_width)


def test_corner_ordering():
    """4. Corner ordering sum/difference helper returns top-left first for arbitrary input."""
    # Points:
    # A (top-left)     : 10, 20
    # B (top-right)    : 90, 25
    # C (bottom-right) : 85, 95
    # D (bottom-left)  : 15, 90
    
    expected = np.array([
        [10, 20],
        [90, 25],
        [85, 95],
        [15, 90]
    ], dtype=np.float32)
    
    # Shuffle points arbitrarily
    shuffled = np.array([
        [85, 95],
        [15, 90],
        [10, 20],
        [90, 25]
    ], dtype=np.float32)
    
    ordered = _order_points(shuffled)
    
    np.testing.assert_array_equal(ordered, expected)


def test_fallback_path_random_noise():
    """5. Fallback path triggers and does not crash on pure random noise as input."""
    # Create random noise image and save to temporary file
    np.random.seed(42)
    noise = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        temp_path = f.name
        cv2.imwrite(temp_path, noise)
        
    try:
        stage = GeometryStage()
        meta = SheetMeta(path=Path(temp_path), date="20.07.2023")
        ctx = {"sheet": meta}
        
        # This should not raise
        ctx_out = stage.run(ctx)
        
        # Verify corners were not found (fallback path was taken)
        assert stage._corners is None
        assert "warped" in ctx_out
        assert ctx_out["warped"] is not None
    finally:
        os.remove(temp_path)
