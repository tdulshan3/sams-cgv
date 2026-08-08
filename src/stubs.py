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


# STUB — owned by M7, delete when their module lands
class DecisionStub(_Stub):
    """Stands in for the present or absent decision and the database write."""

    name = "decision"
    owner = "M7"

    def run(self, ctx: dict) -> dict:
        self.announce()
        ctx["records"] = []
        return ctx


# STUB — owned by M7, delete when their module lands
def parse_students(xml_path: Path) -> list:
    """Stand-in for ``src.io.xml_parser.parse_students``.

    Reads the index and name of every ``<student>`` and nothing else. M7's real
    parser also returns the subject metadata, validates the document and gives
    clear errors on a malformed one — none of which is here.

    It reads the file rather than returning an empty list because the roll is
    what names the saved signature crops. With no students, M6 falls back to
    ``row_<n>.png``, and ``investigate.py`` — which counts a student's samples
    by looking for ``<index>.png`` — reports that everyone has none. Ten lines
    of ``ElementTree`` keeps the rest of the pipeline honest until M7 lands.

    ``.//student`` rather than a fixed path, so it survives the batch element
    changing shape. See BUILD_SPEC.md section 4, deviation 4.
    """
    log.warning("STUB xml_parser: minimal read of %s (owned by M7)", xml_path.name)

    from xml.etree import ElementTree

    from src.models import Student

    try:
        root = ElementTree.parse(xml_path).getroot()
    except (OSError, ElementTree.ParseError) as error:
        log.warning("could not read %s: %s", xml_path, error)
        return []

    students = []
    for node in root.findall(".//student"):
        index = (node.findtext("index") or "").strip()
        name = (node.findtext("name") or "").strip()
        if index:
            students.append(Student(index=index, name=name))
    return students
