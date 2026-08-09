# M9 — Visualisation & QA

Individual contribution notes.

## What I built

Two jobs. The attendance charts behind `infovis.py`, and the quality assurance
that proves the whole system works on all five sheets.

| File | What it does |
|---|---|
| `src/viz/charts.py` | `AttendanceCharts` — timeline, donut, dashboard, heat map, sheet totals, distribution |
| `src/viz/style.py` | One house style every chart in the project uses |
| `tools/run_all_sheets.py` | Wipes the database, runs all five sheets, reports accuracy |
| `tools/make_report_assets.py` | Regenerates every figure in the project in one command |
| `tests/test_charts.py`, `tests/test_e2e.py` | Figure creation, empty states, date ordering, pipeline idempotence |

Data visualisation is one of the module's two named key technologies, so this
is judged on more than whether a bar chart appeared.

## Choosing the chart, not just drawing one

The data is 6 students × 5 sheets = 30 cells, and attendance is high — 25 of
the 30 are present. That shape decided every chart:

* **The heat map is the main one.** Two categories at once, students against
  dates, and at 6 × 5 every cell can carry a label. A lecturer sees the whole
  class and the whole term in one glance and can find both the absent student
  and the difficult sheet.
* **A timeline for one student**, because the question there is order over
  time — did they stop attending, or miss one week?
* **A donut for the percentage**, with the class average drawn alongside. A
  number without a comparison is not a visualisation; 80% means nothing until
  you know the class sits at 83%.
* **The distribution histogram is honest but weak here.** With attendance this
  high it is a spike near 100% and communicates almost nothing. It is kept
  because the spread matters in principle, and the report says plainly that on
  this data it is the least useful of the five.

**Dates must be parsed before they are sorted.** `05.07.2019` sorts before
`21.06.2019` as a string and after it in reality, so every chart parses
`DD.MM.YYYY` first. `tests/test_charts.py` asserts the order explicitly,
because the bug produces a chart that looks entirely plausible.

**Colour.** Red and green alone fails colour-blind readers, so present and
absent differ by position and shape as well as hue, and the palette in
`style.py` is applied once for every chart in the project so the report looks
like one system rather than nine.

## QA — what the runs revealed

`tools/run_all_sheets.py` resets the database, processes all five sheets,
catches failures per sheet and reports accuracy against `data/ground_truth.csv`:

```
Overall Accuracy: 96.7% (29/30 cells correct)
  05.07.2019:  83.3% (5/6)
  12.07.2019: 100.0% (6/6)
  21.06.2019: 100.0% (6/6)
  28.06.2019: 100.0% (6/6)
  31.05.2019: 100.0% (6/6)

Misclassified Cells (Please review):
  - 05.07.2019 / 10009303: Predicted Present, should be Absent
    (cell contains a small stray red pen mark only)
```

Listing the individual failing cell matters more than the percentage. It sends
the group to one exact crop rather than to a number, and `05.07.2019` being the
only imperfect sheet is itself the finding: it is the sheet where a signature
overflows into the row below and contaminates its neighbour.

## Evidence

* `m9_dashboard_example.png` — the full `infovis.py <index>` dashboard
* `m9_class_heatmap.png` — students × dates
* `m9_attendance_distribution.png` — spread across all students
* `m9_accuracy_report.png` — accuracy per sheet with the error count
* `m9_chart_type_choice.png` — the same data as a pie, a bar and a timeline, showing why the choice was made rather than assumed

`python tools/make_report_assets.py` regenerates all 50 figures across the nine
modules in one command, so a late parameter change does not mean re-taking
screenshots by hand.

---

*Integration note: this document was drafted by M1 during integration from the
module's own code and QA output. Review it and rewrite in your own words before
submission.*
