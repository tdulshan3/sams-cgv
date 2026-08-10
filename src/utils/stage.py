"""The OOP backbone of the pipeline.

Every processing step in SAMS, straightening the photo, enhancing it,
binarising it, finding the table, measuring ink, deciding attendance, is a
subclass of :class:`Stage`. That gives us one uniform way to run a step, time
it, report which step failed, and collect the pictures it produced for the
step-by-step viewer the brief asks for.

A stage never prints, never writes files outside ``outputs/``, and never
reaches into another stage. It reads keys out of the context dictionary and
writes its own keys back. The keys are listed in ``BUILD_SPEC.md`` section 6.3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Stage(ABC):
    """One step of the image pipeline.

    Subclasses set :attr:`name` and implement :meth:`run`. Override
    :meth:`figures` to have the step's images appear in the montage.
    """

    name: str = "stage"
    """Short lowercase identifier. Used in logs, timings and error messages."""

    @abstractmethod
    def run(self, ctx: dict) -> dict:
        """Read from ``ctx``, write results back into ``ctx``, return ``ctx``.

        Args:
            ctx: The shared context dictionary. Keys are fixed by the spec.

        Returns:
            The same dictionary, mutated in place. Returning it keeps call
            sites readable and lets a stage be tested on its own.

        Raises:
            Anything. The :class:`~src.pipeline.Pipeline` catches, names the
            failing stage and re-raises. A stage must never swallow its own
            error to keep the run going.
        """

    def figures(self) -> dict[str, np.ndarray]:
        """Images this stage wants shown and saved.

        Returns:
            A mapping of short label to image. Insertion order is the order
            they appear in the montage. The default is nothing, so a stage that
            produces no picture needs no boilerplate.
        """
        return {}

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
