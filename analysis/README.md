# Analysis

Steps that aren't any one model's business.

| Script | Does |
|---|---|
| `preprocess.py` | screening, cleaning, the shared split |
| `assemble_tables.py` | stitches per-model fragments into the numbered tables |
| `representation.py` | **RQ2** — contextual vs lexical, same head, same folds |
| `compare.py` | **the central experiment** — paired McNemar, bootstrap CIs |
| `figures.py` | all seven Chapter Four figures |
| `write_chapters.py` | writes Chapters Four & Five into `docs/` |
| `build_dashboard.py` | builds `results/dashboard.html` |
| `update_thesis.py` | the Chapter One/Three amendments |

```bash
make analysis     # assemble -> representation -> compare -> figures
make chapters
make dashboard
```

Run `assemble_tables` after any model reruns, before `compare`.

`representation.py` lives here rather than in a model because it holds the
classifier and folds fixed and varies only the representation — a cross-cutting
comparison, not a property of one model.

Figure/table numbering: see [NOTES.md](../NOTES.md).
