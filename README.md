# ICT Help Desk Ticket Management

MSc thesis code — Eliud Tuwei, Moi University. Compares **two ways** of handling
three help desk decisions: classification, resolver routing, and recommending
past resolutions.

- **Arm A — one end-to-end model.** One TF-IDF representation, five classifiers
  combined by vote, lexical retrieval. `models/end_to_end/`
- **Arm B — three dedicated models,** each tuned separately on sentence
  embeddings, routing conditioned on the predicted category:
  **B1** `models/classification/`, **B2** `models/routing/`,
  **B3** `models/recommendation/`

Both train and test on the same split, so comparisons are paired.

## Start here

```bash
make            # list every command
make status     # what has been run already
make demo       # six views at http://127.0.0.1:8000/
make all        # rerun the whole pipeline
```

Three views, one server:

| URL | Shows |
|---|---|
| `/` | **Arm A** — one model makes all three decisions |
| `/classification` | **Arm B · B1** — category, on its own |
| `/routing` | **Arm B · B2** — resolver team, on its own |
| `/recommendation` | **Arm B · B3** — prior resolutions, on its own |
| `/cascade` | **Arm B** — B1 → B2 → B3 chained |
| `/dashboard` | the findings, with a present mode (`P` or `?present`) |

Each model page carries a model card — approach, features, representation and
its held-out metrics — so a single model can be presented without the others.

## Results

| Decision | Arm A | Arm B | Winner | p |
|---|---|---|---|---|
| Classification | 0.901 | 0.879 | Arm A | 0.00032 |
| Routing | 0.696 | 0.689 | Arm A | 0.03558 |
| Recommendation | 0.904 | 0.918 | **Arm B** | 0.00011 |

Macro F1, except recommendation (MRR). A split verdict — the cheap lexical
ensemble wins classification, semantic representation wins retrieval.

## Layout

| Folder | What's in it |
|---|---|
| `models/` | Arm A and Arm B's B1–B3 — one folder each, own README |
| `shared/` | data, representations, context blocks, classifiers, metrics |
| `analysis/` | preprocessing, comparison, figures, chapters, dashboard |
| `results/` | tables, figures, `dashboard.html` |
| `docs/` | the thesis, with Chapters Four and Five generated |
| `tools/`, `archive/` | corpus generator; superseded code |

Models are independent: each reads `data/processed/` plus upstream artifacts and
publishes one of its own. Nothing imports another model.

## Two things to know

**The corpus is synthetic.** `data/raw/` stands in for the authorised osTicket
export, pending approval. Comparative findings hold; absolute numbers don't
transfer. Drop in the real export and rerun — everything regenerates.

**Caching.** `results/cache/` holds embeddings and fitted models so reruns take
seconds. `make clean-cache` forces a refit (slow — ~1 hour).

All six pages share one design system (`shared/ui.css`) — one palette, one type
scale, one spacing grid, light and dark. The interface colours and the chart
colours come from the same validated ramps.

See [NOTES.md](NOTES.md) for design decisions and the traps worth knowing.
