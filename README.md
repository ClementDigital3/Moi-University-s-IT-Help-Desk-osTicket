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

To show it on a projector, a phone, or a supervisor's laptop, serve it to the
local network instead:

```bash
make demo-lan   # also at http://<this-machine>:8000/
```

That one prints a generated username and password, because off this machine the
pages include unpublished findings and anyone who can reach the port could
otherwise read them. The password is new each start and shown once; pin your own
with `--auth demo:yourpassword`. This machine itself is exempt, so `make demo`
and `localhost` never ask for anything.

Six views, one server:

| URL | Shows |
|---|---|
| `/` | **Arm A** — one model makes all three decisions |
| `/classification` | **Arm B · B1** — category, on its own |
| `/routing` | **Arm B · B2** — resolver team, on its own |
| `/recommendation` | **Arm B · B3** — prior resolutions, on its own |
| `/cascade` | **Arm B** — B1 → B2 → B3 chained |
| `/tickets` | **the queue** — where each ticket was routed, and every hand-off |
| `/dashboard` | the findings, with a present mode (`P` or `?present`) |

## Where the tickets went

Acting on a suggestion opens a ticket in a small SQLite database
(`results/helpdesk.db`, created on first use — nothing to install). Two tables:
`tickets` holds where each one sits now, `events` holds everything that
happened to it in order.

```bash
make tickets                        # where they are, and which teams pass work on
python -m tools.tickets MU-0007     # one ticket's full history
```

This exists because the decision log alone cannot answer it. That file records
one row per click with no ticket identity anywhere in it, so a ticket that is
rejected, enriched, re-scored and referred leaves five unrelated lines that
nothing joins. "Which team is holding this" was not a hard query against it; it
was an impossible one.

The queue reports a **reassignment rate** against the corpus baseline of 13.5%.
Section 5.4.2 asks future work to evaluate routing on reassignment rather than
classification accuracy — measuring it the same way is what makes the live
figure comparable.

Every suggestion carries **Accept** and **override** controls. Clicking either
records what was suggested against what was chosen, and the running override
rate appears in the session panel — the operational measure Chapter Five asks
for. The log is `results/decisions.jsonl`.

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
