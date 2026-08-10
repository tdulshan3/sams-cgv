"""The pipeline runner.

One photo goes in, one context dictionary comes out. The runner owns the order
of the stages and nothing else: it does not know what deskewing is, it does not
know what ink is. That separation is what lets nine people replace one stage at
a time without touching anything here.

The order is fixed by BUILD_SPEC.md section 6.3:

    geometry -> enhance -> binarize -> table -> ink -> decision
"""

from __future__ import annotations

from src.models import SheetMeta, Student
from src.utils import timing
from src.utils.logging import get_logger
from src.utils.stage import Stage
from src.viz.progress import ProgressViewer

log = get_logger("pipeline")


class PipelineError(RuntimeError):
    """A stage failed.

    The point of this class is the stage name. With nine people's code in one
    line, ``ValueError: operands could not be broadcast`` is useless on its
    own; ``stage 'table' failed: operands could not be broadcast`` says whose
    module to open. The original exception is kept as ``__cause__`` so the full
    traceback survives.

    Args:
        stage: :attr:`~src.utils.stage.Stage.name` of the stage that failed.
        original: The exception it raised.
    """

    def __init__(self, stage: str, original: BaseException) -> None:
        super().__init__(f"stage {stage!r} failed: {original}")
        self.stage = stage
        self.original = original


class Pipeline:
    """Runs a list of stages over one signing sheet.

    Args:
        stages: The stages to run, in order. Whether each one is a real module
            or a stub is the caller's business, not the runner's.
        viewer: Collects the pictures each stage produces. Optional, because a
            unit test has no interest in them.
    """

    def __init__(self, stages: list[Stage], viewer: ProgressViewer | None = None) -> None:
        self.stages = list(stages)
        self.viewer = viewer

    @property
    def stage_names(self) -> list[str]:
        """The names of the stages, in the order they will run."""
        return [stage.name for stage in self.stages]

    def __repr__(self) -> str:
        return f"<Pipeline stages={' -> '.join(self.stage_names)}>"

    def run(self, sheet: SheetMeta, students: list[Student]) -> dict:
        """Run every stage over one sheet.

        Args:
            sheet: Which photo is being processed.
            students: The roll from ``info.xml``, so the decision stage can map
                rows to people.

        Returns:
            The context dictionary, carrying every key the stages wrote. The
            keys are listed in BUILD_SPEC.md section 6.3.

        Raises:
            PipelineError: If any stage raises. The run stops there; a half
                processed sheet must never reach the database looking like a
                real answer.

        Timings for every stage are recorded in :mod:`src.utils.timing` and
        cleared first, so :func:`src.utils.timing.timings` describes this run
        and not the last one.
        """
        ctx: dict = {"sheet": sheet, "students": students}
        timing.reset()
        log.info("processing sheet %s  (%s)", sheet.date, sheet.path.name)
        log.info("stages: %s", " -> ".join(self.stage_names))

        for position, stage in enumerate(self.stages, start=1):
            log.info("[%d/%d] stage %r starting", position, len(self.stages), stage.name)
            try:
                with timing.time_block(stage.name):
                    ctx = stage.run(ctx)
            except Exception as error:
                log.error("stage %r failed: %s", stage.name, error)
                raise PipelineError(stage.name, error) from error
            self._collect_figures(stage)
            log.info("[%d/%d] stage %r done", position, len(self.stages), stage.name)

        log.info("all %d stages finished in %.2f s", len(self.stages), timing.total())
        return ctx

    @staticmethod
    def timings() -> dict[str, float]:
        """Seconds spent in each stage of the run that just finished."""
        return timing.timings()

    @staticmethod
    def duration() -> float:
        """Total seconds across every stage of the run that just finished."""
        return timing.total()

    def _collect_figures(self, stage: Stage) -> None:
        """Hand a stage's pictures to the viewer, if there is one.

        A stage that produces no figure is normal and silent. A stage whose
        ``figures()`` raises is a bug in that stage, and it is reported as
        such rather than being allowed to take the whole run down; the
        pictures are for the report, the attendance is the product.
        """
        if self.viewer is None:
            return
        try:
            figures = stage.figures()
        except Exception:
            log.exception("stage %r failed to produce its figures", stage.name)
            return
        if figures:
            self.viewer.add_all(figures)
