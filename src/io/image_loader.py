"""Image loading utilities."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

from src.config import TARGET_WIDTH


def load_image(path: str | Path) -> np.ndarray:
    """BGR uint8, EXIF rotation applied. FileNotFoundError if missing,
    ValueError if not a decodable image. Both with clear messages.

    Args:
        path: Path to the image file to load.

    Returns:
        The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the file exists but cannot be decoded.
    """
    resolved_path = Path(path)

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Image file not found: {resolved_path}")

    try:
        # Note: none of our 5 real signing-sheet photos carry an EXIF orientation
        # tag (confirmed in BUILD_SPEC.md section 4 / T0), so this code path is
        # a no-op on our actual data, it's here for correctness with other
        # phones/photos, not because our fixtures need it. Don't let its absence
        # look like a bug when you test on the real sheets.
        pil_img = Image.open(resolved_path)
        pil_img = ImageOps.exif_transpose(pil_img)

        # Ensure image is in RGB mode
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Convert to numpy array in RGB
        rgb_array = np.array(pil_img)

        # Convert RGB to BGR for OpenCV consistency
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception as e:
        raise ValueError(f"File exists but is not a decodable image: {resolved_path}") from e


def resize_to_width(bgr: np.ndarray, width: int = TARGET_WIDTH) -> np.ndarray:
    """Downscale only, never upscale, aspect ratio preserved.

    Args:
        bgr: The original image as a NumPy array.
        width: The target width in pixels. Defaults to src.config.TARGET_WIDTH.

    Returns:
        The downscaled image, or the original image if it is already narrower than or equal to `width`.
    """
    h, w = bgr.shape[:2]
    
    # If the image is already narrower than or equal to `width`, return unchanged (never upscale)
    if w <= width:
        return bgr
        
    ratio = width / w
    new_h = int(h * ratio)
    
    # cv2.resize with INTER_AREA
    return cv2.resize(bgr, (width, new_h), interpolation=cv2.INTER_AREA)
