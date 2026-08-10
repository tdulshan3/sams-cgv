# M1: Lead / Integration & Progress Viewer

**T.R.D.T. Dulshan**
CS402.3 Computer Graphics and Visualization, SAMS group coursework

---

## What I built

I owned the parts of the prototype that everybody else plugs into: the
repository, the shared data contracts, the pipeline that joins the stages
together, the three command line programs the marker runs, and the
step-by-step viewer the brief asks for.

Concretely:

| File | What it does |
|---|---|
| `src/models.py` | The five dataclasses every module passes around: `Student`, `SheetMeta`, `Cell`, `InkResult`, `AttendanceRecord` |
| `src/utils/stage.py` | `Stage`, the abstract base class every processing step subclasses |
| `src/config.py` | Every tunable number in the project, in one file, with a block per member |
| `src/utils/logging.py` | One log format for nine people's output, plus a warning collector |
| `src/utils/timing.py` | `@timed` and `time_block`, so the report can quote real seconds |
| `src/viz/progress.py` | `ProgressViewer`, collects, numbers, saves and montages every step image |
| `src/pipeline.py` | `Pipeline`, runs the stages in order, times them, and names the one that failed |
| `src/stubs.py` | A placeholder stage for every module not yet written |
| `src/cli.py` | Shared validation and friendly errors for the three programs |
| `sams.py`, `infovis.py`, `investigate.py` | The three commands the brief fixes |
| `tools/inspect_inputs.py` | Measures the real input data instead of guessing at it |
| `tools/make_fixtures.py` | Throwaway images so five people could start before stage one existed |
| `tools/make_m1_figures.py` | The architecture, montage and timing figures for the report |
| `tests/test_pipeline.py` | The test suite |

## The idea the whole design rests on

Nine people cannot edit one program. So the pipeline is not a program: it is a
list. Every step is a subclass of `Stage` with a single method:

```python
class Stage(ABC):
    name: str = "stage"

    @abstractmethod
    def run(self, ctx: dict) -> dict:
        """Read from ctx, write results back into ctx, return ctx."""

    def figures(self) -> dict[str, np.ndarray]:
        return {}
```

`Pipeline` walks that list. It does not know what deskewing is or what ink is.
It knows the order, it knows how to time a step, and it knows how to say which
step broke. Swapping a placeholder for a real module is one line in `sams.py`:

```python
STAGES = [
    stubs.GeometryStub,   # M2, src.preprocess.deskew
    ...
]
```

Everything else: the summary table, the montage, the step images, the tests,
keeps working untouched. That single decision is what let the other eight
members work in parallel from the first day.

## Techniques and libraries

**Abstract base classes (`abc`).** `Stage` cannot be instantiated. Anyone who
forgets to implement `run` finds out at construction, not three stages into a
run, which matters when the person who wrote the stage is not the person
running it.

**Context dictionary over constructor injection.** Stages could have been wired
to each other directly, but then adding a key would mean editing every stage
downstream of it. One `ctx` dict with keys fixed in the spec means M6 can start
reading `ctx["cells"]` before M5 has written a line.

**OpenCV and NumPy.** Images are `uint8` NumPy arrays in BGR order throughout.
Two conventions do all the work and are stated in `models.py`: colour is BGR
because that is what OpenCV produces, and after binarisation **ink is 255 and
paper is 0**, so ink is the foreground for every morphology operation
downstream.

**Matplotlib for the progress display.** The brief asks for the processing to
be shown as it happens. `ProgressViewer` collects a labelled image from each
stage, writes them as `outputs/steps/<date>/01_original.png` … and lays them
out on one figure whose grid shape adapts to the number of steps.

**Context managers and decorators for timing.** `time_block` wraps each stage
in `Pipeline.run` in a `try/finally`, so a stage that raises is still measured
and still reported.

**`logging` with a single configured handler.** Nine modules call
`get_logger("their_name")` and every line comes out in the same shape:
`[14:22:07] INFO    binarize | otsu threshold = 137`.

**pytest.** 17 tests covering stage ordering, error reporting, step numbering,
colour and channel handling, the summary arithmetic, and every command's exit
code, run with the `Agg` backend so no test ever opens a window.

## Problems and how I solved them

**The spec was wrong about the sheet, and I found out by measuring.** The
project plan assumed a four column table with the signature in column 3 and
three digit indices like `001`. Task T0 was written to check that rather than
trust it. The real sheets have **five** columns, there is a `Title` column of
`Mr`/`Ms` between the student number and the name, so the signature is column
**4**, and the indices are eight digits. Had anyone hard-coded 3, every cell
read would have been the student's name. The measured facts and eight further
deviations are recorded in section 4 of `BUILD_SPEC.md`, which I updated and
committed *before* writing any pipeline code.

**`info.xml` was never supplied, and the brief's own example cannot be
parsed.** Figure 1 of the brief shows the batch element written as `<15>`. An
XML tag may not begin with a digit, so no standard parser will read that
document. I reconstructed the file from the brief's structure plus the printed
student table, carrying the batch as `<batch year="2016.1">`, and documented
exactly what changed and why so M7's parser is not written against a fiction.

**Nobody could start until there was something to process.** M4 through M8 all
need an input image and none of them could wait for M2 and M3. I wrote
`tools/make_fixtures.py`, which fakes the first three stages with the crudest
OpenCV calls that produce something usable, a hand-measured rectangular crop
instead of a perspective transform, plain Otsu instead of adaptive
thresholding. Its docstring says in the first line that it is scaffolding and
when to delete it.

**BGR against RGB.** OpenCV orders colour blue, green, red; Matplotlib reads
red, green, blue. Handing a step image straight to `imshow` turns every blue
pen orange: and it looks deliberate, so it survives review. `_for_display`
converts, squeezes a `(h, w, 1)` array down to 2D and applies a grey colormap,
because otherwise a greyscale sheet renders in purple and yellow.

**A traceback names a line, not a person.** With nine people's code in one run,
`ValueError: operands could not be broadcast` says nothing about whose module
to open. `PipelineError` wraps whatever a stage raises and puts the stage name
in the message: `stage 'table' failed: no horizontal lines found`, while
keeping the original exception as `__cause__` so the full traceback survives.

**Warnings scroll past.** A `WARNING` forty lines above the answer has not been
read. `WarningCollector` captures every warning of a run and the summary block
repeats them under the counts.

**Full size step images are wasteful.** Eight 3024 x 4032 PNGs per sheet is
tens of megabytes that nobody looks at, since each is a few inches wide in the
report. Step images are downscaled to 1400 px wide on the way to disk.

## Evidence

| Where | What it shows |
|---|---|
| `outputs/figures/m1_architecture.png` | The pipeline, one block per stage, with owner and context key |
| `outputs/figures/m1_montage_12.07.2019.png` | The step-by-step montage the brief asks for |
| `outputs/figures/m1_timing.png` | Seconds per stage |
| `outputs/steps/<date>/` | The numbered step images, one folder per sheet, used by everyone |
| `pytest -q` | 17 passing tests |
| `git log --author="Dulshan"` | The commit history, one logical change per commit |

## What is left

`src/stubs.py` still holds seven placeholders. Each disappears as its module is
merged: task T11, and the `v1.0` tag waits for the last one. My part is
tagged `v0.1-m1-core`.
