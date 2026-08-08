# M7 — Decision & Database

Individual contribution notes. Two pages minimum, per the coursework brief:
what you built, the techniques you used and why, the problems you hit, and
the figures in `outputs/` that back it up.

## What I built

The last stage of the pipeline and the only source of data for the two modules
after it. Four separate pieces of work:

| File | What it does |
|---|---|
| `src/io/xml_parser.py` | reads `info.xml` into `Student` objects and the subject metadata |
| `src/detect/presence.py` | maps sheet rows to students, decides present or absent, persists the sheet |
| `src/io/db.py` | the SQLite schema and every query the project runs |
| `tools/seed_db.py` | fake-but-real-shaped data so M8 and M9 could start before the pipeline worked |
| `tools/tune_threshold.py` | sweeps the decision threshold against the hand-labelled ground truth |
| `tools/make_m7_figures.py` | the five report figures, drawn from that same sweep |

The database went first, before any decision logic, because M8 and M9 both read
from it and neither could start until the schema existed. `tools/seed_db.py`
exists for the same reason: it writes thirty rows of invented attendance with
the real schema, the real six students and the real five sheet dates, so the
charts and the signature comparison could be developed against something before
the image pipeline produced anything.

Four tables — `students`, `sheets`, `attendance`, `signatures` — shown in
`m7_er_diagram.png`. Two design points are worth naming. `attendance` is unique
on `(student_index, sheet_id)` and every write is `INSERT OR REPLACE`, so
re-running `sams.py` on a sheet updates the verdict instead of adding a second
one; the row count after processing all five sheets twice is still 30, which is
what `tests/test_db.py::test_processing_a_sheet_twice_does_not_duplicate` pins
down. And every query binds its values as `?` parameters, which
`test_sql_injection_in_an_index_changes_nothing` demonstrates by passing
`10000409'; DROP TABLE attendance; --` as a student index and asserting the
table is still there afterwards — that string reaches the database whenever a
user mistypes an argument to `infovis.py`.

## Techniques and libraries

**`xml.etree.ElementTree`, searching rather than walking.** The brief's Figure 1
shows the batch element as `<15>`. XML tag names may not begin with a digit, so
that document is not well-formed and no standard parser will read it at all —
our reconstructed file carries `<batch year="2016.1">` instead. Rather than
depend on either shape, the parser finds students with `.//student` and the
subject with `.//subject`, which survives the nesting changing again. Indices
are kept as strings throughout; `int()` anywhere near one destroys a leading
zero and turns `"007"` into a different student.

**SQLite through `sqlite3`, with context-managed connections.** The whole
database is one file with no server to install, which matters for a prototype a
marker has to run. `Database.connect()` is a `@contextmanager` that commits on
success and always closes, so no query in the project can leak a handle.

**Ordering dates in SQL rather than in Python.** Sheet dates are stored as they
are printed, `DD.MM.YYYY`. Sorted as text that puts `05.07.2019` before
`21.06.2019`, and every chart of attendance over time would draw the term in
the wrong order. `ORDER BY substr(sheet_date,7,4), substr(sheet_date,4,2),
substr(sheet_date,1,2)` slices the string back into year, month, day, which
sorts correctly without adding a second date column.

**Choosing the threshold by measurement.** `INK_RATIO_THRESHOLD` decides who
gets marked absent, so it is the one number in this module that must not be
guessed. `tools/tune_threshold.py` runs the real pipeline over all five sheets,
joins M6's measurements for each of the thirty signature cells against
`data/ground_truth.csv`, and sweeps the threshold across its entire range.

## Problems and how I solved them

**The sweep has no peak, it has a plateau.** With thirty cells, every threshold
between the highest genuinely-blank cell (0.0173) and the lowest real signature
(0.0546) scores identically, because no cell lies between them. Taking the
first best-scoring value would put the threshold hard against one cell's
measurement. I take the midpoint of the widest plateau instead — 0.0359, which
is the value furthest from being wrong about any cell we have seen.

**The plateau lied the first time I looked at it.** Swept with the full rule,
the plateau ran all the way down to zero, which would have suggested a
threshold of nearly nothing was fine. It was not: `MIN_STROKE_LENGTH` was
quietly rejecting the faint cell on its own, and a threshold chosen there would
have been resting on a different feature entirely. The tool now runs two
sweeps — ink ratio alone, which chooses the threshold, and the full rule, which
shows what the other features buy — and `m7_threshold_sweep.png` plots both.
Ink alone reaches 93.3%; the full rule reaches 96.7%.

**Ink is not the same thing as a signature.** Two cells were known in advance to
hold ink that means *absent* (BUILD_SPEC.md §4 deviation 7), and they are the
whole reason the rule is not one number:

- **21.06.2019 / 10009306** — the lecturer marked the student absent by ruling a
  line across the box and writing `ab` on it. It covers 74% of the cell, where
  the largest genuine signature in the data covers 37%. `MAX_INK_RATIO = 0.55`
  rejects it. This is the one rule here fitted to a single example, so I have
  flagged it as such in `config.py` rather than presenting it as settled: a
  signature is strokes on paper and not a filled box, so an upper bound is the
  right *shape* of rule, but one supporting cell is not evidence that 0.55 is
  the right number.
- **05.07.2019 / 10009303 — still wrong, and the interesting one.** The cell
  holds a small stray red pen mark on otherwise blank paper. It should read as
  near-zero ink. It reads 0.274, because M6's ink mask marks a large region of
  blank paper as ink; opening the crop and its mask side by side makes this
  obvious. No threshold on any feature M6 supplies separates it from a genuine
  signature — its stroke length (192) and fill ratio (0.77) both sit inside the
  present distribution. **This failure is in the ink mask, not in the decision
  rule, and no decision rule can fix it.** I raised it with M6 rather than
  fitting a rule around one bad measurement.

**A missing row must not shift every student up one.** The row-to-student
mapping is positional, which the spec confirms is safe here because the XML was
transcribed off the sheets in row order. The failure mode is the interesting
part: if M5 finds five rows where the roll has six, zipping the two lists
blindly attaches five students to the wrong rows and produces six confident,
wrong records. `map_rows_to_students` maps only what it can and warns loudly,
which is the failure worth having.

**M6 could not name its own output files.** The spec has M6 saving crops as
`outputs/cells/<date>/<index>.png` for M8, but at that point in the pipeline
nothing knows the index — the mapping is this module's job, and it runs after.
M6 saves `row_<n>.png`; `DecisionStage` renames each to `<index>.png` once the
mapping is known, which is the first point in the run where both facts are in
hand. `investigate.py` finds all five samples per student as a result.

**Accuracy, reported both ways.** 29/30 = **96.7%** over all thirty cells;
28/28 = **100%** over the twenty-eight that exclude the two cells known in
advance to hold non-signature ink. Both numbers are in
`m7_confusion_matrix.png` and neither is quoted on its own — the second alone
would be flattering, the first alone hides which failure belongs to which
module. Every error is a false *present*: no student who signed is ever marked
absent, which is the direction to err in if attendance affects eligibility.

## Evidence

| Figure | What it shows |
|---|---|
| `m7_ink_ratio_distribution.png` | where signed and unsigned cells actually sit, the threshold, and the two ink-but-absent cells named on the strip below |
| `m7_threshold_sweep.png` | accuracy at every candidate threshold, ink alone against the full rule, with the plateau shaded |
| `m7_confusion_matrix.png` | the verdicts, counted with and without the two known cells |
| `m7_accuracy_per_sheet.png` | the single error, and which sheet it is on |
| `m7_er_diagram.png` | the four tables, their keys and their uniqueness constraints |

Reproduce all of it from a clean checkout with:

```bash
python tools/tune_threshold.py      # runs all five sheets, caches the measurements
python tools/make_m7_figures.py     # draws the five figures from that cache
pytest tests/test_decision.py tests/test_db.py -q
```
