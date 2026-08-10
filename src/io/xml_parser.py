"""Reading ``info.xml``, the roll, and who the subject belongs to.

The coursework brief shows this file in its Figure 1 as
``<info><students><student>``, with the batch written as ``<15>``. Neither is
what we actually hold. A tag may not begin with a digit, so Figure 1 is not
well-formed XML and no standard parser reads it at all; the file committed at
``data/info.xml`` carries the batch as ``<batch year="2016.1">`` and nests the
students as ``nsbm/students/batches/batch/student``. See BUILD_SPEC.md §4
deviation 4.

That history is the reason this module never walks a fixed path. It searches
with ``.//student`` and ``.//subject``, so the batch element can change shape
again: or gain another level of nesting, without breaking the parse. The one
thing it does insist on is that each student carries an ``index`` and a
``name``; a roll with a nameless student in it is a broken input, not something
to paper over.

**An index is a string.** ``"10000409"``, and on another sheet it might be
``"007"``. ``int()`` anywhere near one silently destroys a leading zero and
turns a student into a different student.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from src import config
from src.models import Student
from src.utils.logging import get_logger

log = get_logger("xml")

STUDENT_PATH = ".//student"
"""Searched from the root, so nesting above a student does not matter."""

SUBJECT_PATH = ".//subject"

SUBJECT_FIELDS = ("code", "name", "degree", "lecturer")
"""Child tags read out of ``<subject>``. Any that is absent comes back ``""``."""


def _text(element: ET.Element, tag: str) -> str:
    """The stripped text of one child tag, or ``""`` when it is absent or empty."""
    child = element.find(tag)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def _read_tree(path: str | Path) -> ET.Element:
    """Parse the file into its root element, with errors a human can act on.

    Args:
        path: Path to ``info.xml``.

    Returns:
        The root element.

    Raises:
        FileNotFoundError: If there is no file there.
        ValueError: If the file is not well-formed XML. The message keeps the
            line and column the parser choked on, because that is the only
            thing that makes a broken XML file quick to fix.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"No student list found at {resolved}")
    try:
        return ET.parse(resolved).getroot()
    except ET.ParseError as error:
        raise ValueError(f"{resolved} is not well-formed XML: {error}") from error


def parse_info(path: str | Path = config.INFO_XML) -> tuple[list[Student], dict]:
    """Read the whole file: the students, and the subject they are enrolled on.

    Args:
        path: Path to ``info.xml``. Defaults to the committed one.

    Returns:
        ``(students, meta)``. ``students`` is in document order, which is sheet
        row order: the XML was transcribed off the sheets row by row, so
        student *n* is row *n*. ``meta`` holds ``subject_code``,
        ``subject_name``, ``degree`` and ``lecturer``, each ``""`` if the file
        does not carry it.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the XML is malformed, holds no ``<student>`` at all, or
            has a student with no index or no name.
    """
    root = _read_tree(path)

    subject = root.find(SUBJECT_PATH)
    meta = {
        f"subject_{field}" if field in ("code", "name") else field: (
            _text(subject, field) if subject is not None else ""
        )
        for field in SUBJECT_FIELDS
    }

    students: list[Student] = []
    for position, element in enumerate(root.iterfind(STUDENT_PATH)):
        index = _text(element, "index")
        name = _text(element, "name")
        if not index:
            raise ValueError(
                f"{path}: student {position + 1} has no <index>; "
                "every student needs one, it is how attendance is keyed"
            )
        if not name:
            raise ValueError(f"{path}: student {index} has no <name>")
        students.append(Student(index=index, name=name))

    if not students:
        raise ValueError(
            f"{path}: no <student> elements found. Expected them under "
            "nsbm/students/batches/batch/, see BUILD_SPEC.md section 4."
        )

    log.info(
        "%d students read for %s", len(students), meta.get("subject_code") or "an unnamed subject"
    )
    return students, meta


def parse_students(path: str | Path = config.INFO_XML) -> list[Student]:
    """The roll alone, in sheet row order.

    This is the function ``sams.py`` and ``src/cli.py`` import (BUILD_SPEC.md
    §6.5); everything that also wants the subject calls :func:`parse_info`.
    """
    students, _ = parse_info(path)
    return students
