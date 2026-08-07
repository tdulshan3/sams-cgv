"""Geometry and perspective correction stage."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    CANNY_LOW,
    CANNY_HIGH,
    MIN_SHEET_AREA_RATIO,
    MAX_SKEW_CORRECTION_DEG,
    BORDER_TRIM_PX,
)
from src.io.image_loader import load_image, resize_to_width
from src.utils.stage import Stage


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 arbitrary points into top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # top-left = smallest sum (x+y)
    # bottom-right = largest sum (x+y)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # diff computes (y-x) when axis=1 on an array of (x,y)
    # top-right = smallest difference (y-x)
    # bottom-left = largest difference (y-x)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def find_sheet_corners(bgr: np.ndarray) -> np.ndarray | None:
    """(4, 2) float32, ordered top-left, top-right, bottom-right, bottom-left.
    None when not found — that is a normal outcome, not an error."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    full_area = bgr.shape[0] * bgr.shape[1]

    # Largest first, so the page outline is tried before anything inside it.
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        perimeter = cv2.arcLength(contour, True)

        # Tolerance loop starting near 0.02 * perimeter to find exactly 4 points
        for factor in np.linspace(0.01, 0.1, 100):
            epsilon = factor * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                # Reject anything too small to be the sheet. Without this the
                # first four-sided thing found wins, which on these photos is
                # the student table or a fragment of it — the warp then crops
                # to a sliver and every later stage sees almost nothing.
                if cv2.contourArea(approx) / full_area >= MIN_SHEET_AREA_RATIO:
                    pts = approx.reshape((4, 2)).astype(np.float32)
                    # approxPolyDP returns the corners in traversal order, which
                    # may start anywhere and run either way round. four_point_warp
                    # unpacks them as tl, tr, br, bl, so they have to be sorted
                    # first or the transform mirrors the sheet.
                    return _order_points(pts)
                break

    return None



def four_point_warp(bgr: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective transform to a straight top-down view."""
    tl, tr, br, bl = corners

    # Calculate output width as max of top or bottom edge lengths
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = int(max(width_top, width_bottom))

    # Calculate output height as max of left or right edge lengths
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = int(max(height_left, height_right))

    # Construct the destination points to map to
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(bgr, M, (max_width, max_height))


def estimate_skew_angle(grey: np.ndarray) -> float:
    """Small residual rotation in degrees, from Hough lines or minAreaRect."""
    edges = cv2.Canny(grey, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    
    if lines is None:
        return 0.0
        
    angles = []
    for line in lines:
        pts = line[0] if line.ndim == 2 else line
        x1, y1, x2, y2 = pts
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Normalize angle to [-90, 90] range
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
            
        # Filter for near-horizontal segments (+/- 30 degrees)
        if -30.0 <= angle <= 30.0:
            angles.append(angle)
            
    if not angles:
        return 0.0
        
    return float(np.median(angles))


def _trim_border(img: np.ndarray, px: int = BORDER_TRIM_PX) -> np.ndarray:
    """Trim a fixed number of pixels from all four edges."""
    h, w = img.shape[:2]
    if h <= 2 * px or w <= 2 * px:
        return img
    return img[px:h-px, px:w-px]


class GeometryStage(Stage):
    """Initial fallback implementation of the geometry stage."""
    name = "geometry"

    def run(self, ctx: dict) -> dict:
        sheet = ctx["sheet"]
        
        # 1. Load the image
        bgr_full = load_image(sheet.path)
        
        # 2. Resize and store
        bgr = resize_to_width(bgr_full)
        ctx["bgr"] = bgr
        self._original = bgr.copy()
        
        # Compute edges for both estimation and figures
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self._edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        
        # 3. Attempt to find corners and warp
        corners = find_sheet_corners(bgr)
        self._corners = corners
        
        if corners is not None:
            warped = four_point_warp(bgr, corners)
        else:
            # Fallback path: use full image and rotate by estimated skew angle
            # Reusing the computed gray edges for skew estimation
            angle = estimate_skew_angle(gray)
            
            # Clamp the result to +/- MAX_SKEW_CORRECTION_DEG
            angle = max(-MAX_SKEW_CORRECTION_DEG, min(MAX_SKEW_CORRECTION_DEG, angle))
            
            h, w = bgr.shape[:2]
            center = (w / 2, h / 2)
            
            # Get OpenCV rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Expand the canvas so nothing is cut off
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            warped = cv2.warpAffine(bgr, M, (new_w, new_h))
            
        warped = _trim_border(warped)
        
        # 4. Store the final result
        ctx["warped"] = warped
        self._warped = warped.copy()
        
        # 5. Return ctx
        return ctx

    def figures(self) -> dict[str, np.ndarray]:
        if not hasattr(self, "_original"):
            return {}
            
        overlay = self._original.copy()
        if getattr(self, "_corners", None) is not None:
            pts = self._corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            for pt in self._corners:
                cv2.circle(overlay, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
                
        return {
            "original": self._original,
            "edge map": self._edges,
            "corner overlay": overlay,
            "warped result": self._warped,
        }
