"""Placeholder stages, so the pipeline runs end to end from the first hour.

Nine people cannot all wait for stage one. Every module that is not written yet
has a stub here that writes a type-correct value into the context, logs loudly
that it is a stub, and gets out of the way. ``sams.py`` therefore reaches its
summary table on day one, and every member can see their own stage slot in the
montage before they have written a line.

Each stub is deleted the moment its real module is merged — see BUILD_SPEC.md
section 8. Nothing here may survive to submission::

    grep -r "STUB" src/     # must print nothing before tagging

Image stubs read from ``data/fixtures/``, produced by ``tools/make_fixtures.py``.
That folder is not committed, so every stub also has a fallback that keeps a
fresh clone running rather than crashing on a missing file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src import config
from src.utils.logging import get_logger
from src.utils.stage import Stage

log = get_logger("stubs")


class _Stub(Stage):
    """Shared behaviour: announce loudly that this is not the real thing."""

    owner: str = "M?"
    """Which member owns the module this stub is standing in for."""

    def announce(self) -> None:
        """Log the line every stub must print, exactly once per run."""
        log.warning("STUB %s: returning placeholder data (owned by %s)", self.name, self.owner)

    def _fixture(self, filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
        """Load one bootstrap fixture, or ``None`` with a helpful warning."""
        path: Path = config.FIXTURES / filename
        if not path.is_file():
            log.warning(
                "fixture %s is missing — run `python tools/make_fixtures.py`",
                path.relative_to(config.ROOT),
            )
            return None
        image = cv2.imread(str(path), flags)
        if image is None:
            log.warning("fixture %s could not be decoded by OpenCV", path)
        return image
