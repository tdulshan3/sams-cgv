"""Step-by-step progress viewer.

The brief asks for the image processing to be *shown* as it happens, greyscale,
binarisation and everything else, not just for a final answer. This class is
how that happens. Each stage hands its pictures over as it finishes, the viewer
keeps them in order, and at the end of the run it writes them out and puts them
on screen as one montage.

Nothing here knows what a signing sheet is. It takes labelled images and shows
them, which is why it can serve all nine modules.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from src import config
from src.utils.logging import get_logger

log = get_logger("progress")


@dataclass(frozen=True)
class Step:
    """One labelled picture from the pipeline, in the order it arrived."""

    name: str
    """Short label, e.g. ``"binarize"``. Becomes part of the filename."""

    image: np.ndarray
    """The picture itself, in whatever form the stage produced it."""

    cmap: str | None = None
    """Matplotlib colormap for single channel images. ``None`` picks a default."""


class ProgressViewer:
    """Collects the images a pipeline run produces, in order.

    Args:
        sheet_date: Identifies the run. Used as the output folder name.
        show: Whether the montage is allowed to open a window.
        save: Whether step images are allowed to be written to disk.
        steps_root: Where ``<sheet_date>/`` is created. Defaults to
            ``outputs/steps``; tests point it at a temporary folder.
    """

    def __init__(
        self,
        sheet_date: str,
        show: bool = True,
        save: bool = True,
        steps_root: Path | None = None,
    ) -> None:
        self.sheet_date = sheet_date
        self.show = show
        self.save = save
        self.steps_root = Path(steps_root) if steps_root is not None else config.STEPS
        self._steps: list[Step] = []

    @property
    def steps(self) -> list[Step]:
        """The steps collected so far, in insertion order."""
        return list(self._steps)

    @property
    def output_dir(self) -> Path:
        """Folder this run's step images are written to."""
        return self.steps_root / self.sheet_date

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return (
            f"<ProgressViewer sheet={self.sheet_date!r} "
            f"steps={len(self._steps)} show={self.show} save={self.save}>"
        )

    def add(self, name: str, image: np.ndarray, cmap: str | None = None) -> None:
        """Register one pipeline step image.

        Args:
            name: Short label for the step.
            image: The picture. Colour images are BGR, as OpenCV produces them.
            cmap: Colormap for a single channel image.

        Raises:
            ValueError: If ``image`` is not a NumPy array with 2 or 3 dimensions.
                Catching this here means a mistake surfaces at the stage that
                made it, not three steps later in Matplotlib.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"step {name!r}: expected a numpy array, got {type(image).__name__}"
            )
        if image.ndim not in (2, 3):
            raise ValueError(
                f"step {name!r}: expected a 2D or 3D array, got shape {image.shape}"
            )

        self._steps.append(Step(name=name, image=image, cmap=cmap))
        log.info("[%d] %s … collected  %s", len(self._steps), name, image.shape)

    def add_all(self, figures: dict[str, np.ndarray]) -> None:
        """Register every figure a stage produced, preserving its order."""
        for name, image in figures.items():
            self.add(name, image)

    # -- saving ------------------------------------------------------------

    def save_all(self) -> list[Path]:
        """Write every step to ``outputs/steps/<date>/NN_name.png``.

        Files are numbered from ``01`` in the order the steps were added, so
        the folder reads top to bottom as the journey of the photo and can be
        pasted straight into the report.

        Returns:
            The paths written, in order. Empty when ``save`` is ``False``.
        """
        if not self.save:
            log.debug("saving disabled, skipping %d steps", len(self._steps))
            return []
        if not self._steps:
            log.warning("no steps to save for sheet %s", self.sheet_date)
            return []

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Clear the folder first. Step images are numbered by position, so when
        # the pipeline gains or loses a stage the new run writes a different
        # set of names and the old ones survive beside them. After the stubs
        # were replaced this left 02_warped, 03_grey and 04_binary sitting
        # among the seventeen real steps, three pictures of a pipeline that no
        # longer exists, in the folder the report takes its screenshots from.
        for stale in self.output_dir.glob("*.png"):
            stale.unlink()

        written: list[Path] = []
        for number, step in enumerate(self._steps, start=1):
            path = self.output_dir / f"{number:02d}_{self._slug(step.name)}.png"
            image = self._for_file(step.image)
            if not cv2.imwrite(str(path), image):
                raise OSError(f"could not write step image {path}")
            written.append(path)
            log.info(
                "[%d/%d] %s … saved  %s",
                number,
                len(self._steps),
                step.name,
                path.name,
            )
        return written

    @staticmethod
    def _slug(name: str) -> str:
        """Filename-safe version of a step label."""
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "step"

    @staticmethod
    def _as_uint8(image: np.ndarray) -> np.ndarray:
        """Convert any sensible array to ``uint8`` without changing what it shows.

        Stages produce masks as ``bool``, distance maps as ``float`` in
        ``[0, 1]`` and everything else as ``uint8``. The viewer accepts all
        three rather than making eight other people remember to convert.
        """
        if image.dtype == np.uint8:
            return image
        if image.dtype == bool:
            return (image.astype(np.uint8)) * 255
        if np.issubdtype(image.dtype, np.floating):
            finite = image[np.isfinite(image)]
            top = float(finite.max()) if finite.size else 0.0
            scale = 255.0 if top <= 1.0 else 1.0
            return np.clip(np.nan_to_num(image) * scale, 0, 255).astype(np.uint8)
        return np.clip(image, 0, 255).astype(np.uint8)

    def _for_file(self, image: np.ndarray) -> np.ndarray:
        """Downscale and normalise one step image for writing to disk.

        Full size step images are 3024 x 4032. Eight of those per sheet is tens
        of megabytes nobody looks at, in the report each one is a few inches
        wide. They are written at :data:`src.config.STEP_IMAGE_MAX_WIDTH`.

        Channel order is left exactly as it arrived, because ``cv2.imwrite``
        expects BGR and that is what the stages produce.
        """
        prepared = self._as_uint8(image)
        width = prepared.shape[1]
        limit = config.STEP_IMAGE_MAX_WIDTH
        if width > limit:
            scale = limit / width
            prepared = cv2.resize(
                prepared,
                (limit, max(1, int(round(prepared.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return prepared

    # -- display -----------------------------------------------------------

    def _for_display(self, step: Step) -> tuple[np.ndarray, str | None]:
        """Prepare one step for Matplotlib.

        OpenCV and Matplotlib disagree about channel order. OpenCV keeps
        colour as blue, green, red; Matplotlib reads it as red, green, blue.
        Handing a BGR array straight to ``imshow`` swaps the two ends of the
        spectrum, so a blue pen renders orange and every step image in the
        report is wrong in a way that looks deliberate.

        Single channel images have the opposite problem: without a colormap
        Matplotlib applies its default ``viridis`` and a greyscale sheet comes
        out purple and yellow.

        Returns:
            ``(array, cmap)`` ready for ``imshow``.
        """
        image = self._as_uint8(step.image)

        # A 3D array with one channel is still a greyscale image.
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]

        limit = config.STEP_IMAGE_MAX_WIDTH
        if image.shape[1] > limit:
            scale = limit / image.shape[1]
            image = cv2.resize(
                image,
                (limit, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )

        if image.ndim == 2:
            return image, step.cmap or "gray"
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA), None
        raise ValueError(
            f"step {step.name!r}: cannot display an image with "
            f"{image.shape[2]} channels"
        )

    # -- montage -----------------------------------------------------------

    @staticmethod
    def _grid(count: int) -> tuple[int, int]:
        """Rows and columns for ``count`` tiles.

        The grid widens up to :data:`src.config.MONTAGE_MAX_COLS` and then
        wraps, so four steps sit in one row and eight in two, rather than
        eight unreadable slivers side by side.
        """
        columns = max(1, min(config.MONTAGE_MAX_COLS, count))
        rows = math.ceil(count / columns)
        return rows, columns

    def build_montage(self) -> Figure:
        """Lay every collected step out on one Matplotlib figure.

        Returns:
            The figure. The caller decides whether to show it, save it or both,
            which is what lets one montage serve both the live window and
            ``outputs/figures/m1_montage_<date>.png``.

        Raises:
            RuntimeError: If no steps were collected. A montage of nothing is a
                bug in the pipeline, not a picture.
        """
        if not self._steps:
            raise RuntimeError(
                f"no steps collected for sheet {self.sheet_date}, nothing to show"
            )

        rows, columns = self._grid(len(self._steps))
        tile_w, tile_h = config.MONTAGE_FIGSIZE_PER_TILE
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(columns * tile_w, rows * tile_h),
            dpi=config.FIGURE_DPI,
        )
        flat = np.atleast_1d(np.asarray(axes, dtype=object)).ravel()

        for position, axis in enumerate(flat):
            if position >= len(self._steps):
                axis.axis("off")
                continue
            step = self._steps[position]
            image, cmap = self._for_display(step)
            axis.imshow(image, cmap=cmap)
            axis.set_title(f"{position + 1:02d}  {step.name}", fontsize=9)
            axis.axis("off")

        figure.suptitle(
            f"SAMS processing steps, sheet {self.sheet_date}", fontsize=12
        )
        figure.tight_layout()
        return figure

    def show_montage(self) -> None:
        """Open the montage window, unless ``show`` is off."""
        if not self.show:
            log.debug("display disabled, not showing the montage")
            return
        self.build_montage()
        plt.show()

    def save_montage(self, path: Path) -> Path:
        """Write the montage to ``path`` for the report."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure = self.build_montage()
        figure.savefig(path, dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(figure)
        log.info("montage saved to %s", path)
        return path
