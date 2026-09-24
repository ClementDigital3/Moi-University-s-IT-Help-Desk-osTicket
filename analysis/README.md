# Analysis

Steps that are not any one model's business.

| Script | Does | Produces |
|---|---|---|
| `preprocess.py` | screening, cleaning, normalisation, the one shared stratified split | `data/processed/`, Tables 4.1–4.2 |
| `assemble_tables.py` | stitches per-model `_m*.csv` fragments into the numbered tables and the paired-prediction file | Tables 4.3, 4.6, 4.7, 4.15 |
| `representation.py` | **Objective 2 / RQ2** — contextual vs lexical, same head, same folds | Table 4.4 |
| `compare.py` | **the central experiment** — Arm A vs Arm B, paired McNemar, bootstrap CIs, operational comparison | Tables 4.8–4.10, 4.16 |
| `dashboard_data.py` | collects every result table into one JSON payload | (inspectable: run it directly) |
| `build_dashboard.py` | embeds that payload in `dashboard_template.html` | `results/dashboard.html` |
| `figures.py` | all seven Chapter Four figures at 300 dpi | `results/figures/` |
| `write_chapters.py` | writes Chapters Four and Five into a copy of the thesis | `docs/...(complete).docx` |
| `update_thesis.py` | the Chapter One / Three amendments (Objective 3, RQ3, Table 3.2) | `docs/...rev1....docx` |

`representation.py` deliberately lives here rather than inside a model: it holds
the classifier and folds fixed and varies only the representation, so any
difference is attributable to the representation alone. That is a cross-cutting
comparison, not a property of one model.

Run `assemble_tables.py` after any model is re-run, and before `compare.py`.

## The dashboard

`build_dashboard.py` produces **one self-contained HTML file** — data embedded, no
server, no CDN, no network — so it survives being emailed or opened from a USB
stick in a presentation room.

The data collection (`dashboard_data.py`) is deliberately separate from the page
(`dashboard_template.html`) so the data contract is inspectable on its own: run
the collector directly to dump exactly what the dashboard draws from.

Three things in it are deliberate rather than cosmetic:

- **Each decision keeps its own axis and its own metric name.** Macro F1 and MRR
  are not the same quantity, so they are never plotted on a shared scale.
- **The ablation panels enforce a minimum y-span.** Without it, a 0.0025 change
  fills the panel and reads as a dramatic climb — the opposite of the finding.
- **Significance is shown with a glyph and the p-value, not colour alone.**

The chart palette is the validated default: categorical slots 1–2 (blue/orange)
for the two arms, passing the colourblind-separation and contrast gates in both
light and dark mode; a single blue ramp for the confusion heatmap; grey for
de-emphasised marks, with the value always printed beside the bar in text ink.

## Figure and table numbering

The CSV filenames are historical (`t410_`, `t44_`…) and do **not** match the
numbers in the thesis. `write_chapters.py` assigns the Chapter Four numbers by
order of appearance and keeps every cross-reference consistent; it is the single
source of truth for numbering. Don't renumber the CSVs to match.
