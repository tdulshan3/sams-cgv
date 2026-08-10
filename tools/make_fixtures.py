"""Bootstrap fixtures, temporary scaffolding, delete once M2 to M4 land.

M4, M5, M6, M7 and M8 all need an input image to develop against, and none of
them can wait for the real geometry and enhancement stages to be finished. This
script fakes those stages with the crudest OpenCV calls that produce something
usable: a hand-measured rectangular crop instead of a perspective transform,
plain ``cvtColor`` instead of shadow removal, plain Otsu instead of an adaptive
threshold.

**This is not the pipeline.** Nothing here is imported by ``src/``. The moment
M2, M3 and M4 are merged, these fixtures stop being the truth and this file is
deleted along with ``data/fixtures/``.

Written by M1 for task T4. Usage:

    python tools/make_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Scripts in tools/ are run as `python tools/<name>.py` from the repository
# root, which puts tools/ on sys.path and not the root. Add the root so `src`
# imports the same way it does everywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger("fixtures")

# Hand-measured on the first sheet, as fractions of width and height so the
# numbers survive a different photo resolution. Crude on purpose.
SHEET_CROP = (0.03, 0.25, 0.92, 0.60)
"""``(left, top, right, bottom)`` of the printed area, fractions of the image."""

CELL_CROP = (0.645, 0.437, 0.833, 0.460)
"""``(left, top, right, bottom)`` of one signed signature cell, same units.

Row 3 of the first sheet, student 10009302, a clear blue signature whose
stroke runs past the right hand border, which is the awkward case M6 has to
cope with anyway.
"""


def _crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crop by fractional ``(left, top, right, bottom)``."""
    height, width = image.shape[:2]
    left, top, right, bottom = box
    return image[
        int(top * height): int(bottom * height),
        int(left * width): int(right * width),
    ]


def _first_sheet() -> "tuple[str, np.ndarray]":
    """Load the first sheet in the folder, or exit with a friendly message."""
    sheets = sorted(config.SHEETS.glob("*.png"))
    if not sheets:
        print(f"error: no sheets found in {config.SHEETS}")
        raise SystemExit(2)

    path = sheets[0]
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"error: OpenCV could not read {path}")
        raise SystemExit(2)
    return path.name, image


def main() -> int:
    """Write the four fixture images. Returns a process exit code."""
    config.ensure_dirs()
    name, bgr = _first_sheet()
    log.info("fixtures from %s  %s", name, bgr.shape)

    warped = _crop(bgr, SHEET_CROP)
    grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cell = _crop(bgr, CELL_CROP)

    written = {
        "warped.png": warped,
        "grey.png": grey,
        "binary.png": binary,
        "cell_sample.png": cell,
    }
    for filename, image in written.items():
        path = config.FIXTURES / filename
        cv2.imwrite(str(path), image)
        log.info("wrote %s  %s", path.relative_to(config.ROOT), image.shape)

    print()
    print("Bootstrap fixtures written to data/fixtures/.")
    print("Temporary scaffolding, delete once M2, M3 and M4 are merged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
