# The end-to-end model

**The comparison baseline.** Chapter Four calls this **Arm A**; models 1–3
together are **Arm B**.

One model, one representation, every decision — the integrated treatment the
thesis sets against the task-specific one (the integration gap, Section 2.11).

## Architecture

```
            shared TF-IDF, fitted ONCE
                      │
   ┌──────┬───────────┼───────────┬──────────┐
  MNB   LogReg   Linear SVM      RF      XGBoost     ← each fitted once, probabilities cached
   └──────┴───────────┼───────────┴──────────┘
                      │
          combined decision (soft / weighted / majority vote)
                      │
        ┌─────────────┴─────────────┐
   category                  resolver team

   + TF-IDF cosine retrieval for recommendation
```

Every branch is fitted exactly once per task and its probabilities cached, so
the individual results and the combined result come from a single fitting pass
rather than separate experiments. Re-running with a different combination rule
costs no refitting at all.

## Run

```bash
.venv/bin/python -m models.end_to_end.run
```

Delete `results/cache/probas_*.npz` to refit the branches.

## Current result

| Task | Best branch | Combined | Gain from combining |
|---|---|---|---|
| Ticket classification | Linear SVM 0.8990 | **0.9009** (soft vote) | +0.0019 |
| Resolver routing | Multinomial NB 0.7046 | 0.6958 (majority vote) | −0.0088 |

Recommendation: Top-1 0.8745, MRR 0.9036.

Note the second row: on routing, **combining was worse than the best single
branch**. Voting is not free, and the thesis reports this rather than quietly
selecting the best number.

## How it fares against the task-specific models

| Decision | Winner | p (McNemar, exact) |
|---|---|---|
| Ticket classification | End-to-end (+0.0216 macro F1) | 0.00032 |
| Resolver routing | End-to-end (+0.0070) | 0.03558 |
| Resolution recommendation | Task-specific (+0.0147 MRR) | 0.00011 |

A split verdict, and the most useful result in the study: the cheap lexical
ensemble wins the classification decisions, while semantic representation wins
retrieval. Neither architecture dominates, which is precisely what Section 3.11
asked to be tested rather than assumed.

---

# The prototype

A working demo of this model, for the defence. Self-contained: Python standard
library only, no web framework, no CDN, no network access. It runs on a laptop
with the wifi off.

```bash
.venv/bin/python -m models.end_to_end.export     # fit and persist the bundle (~90 s, once)
.venv/bin/python -m models.end_to_end.serve      # then open http://127.0.0.1:8000/
.venv/bin/python -m models.end_to_end.serve --port 8080 --open
```

`export.py` fits the five branches and the retrieval index **on the training
partition only** and writes `results/deploy/end_to_end.joblib` (34 MB). `serve.py`
loads it once at startup, so every prediction is instant.

### What it shows

For a ticket you type in, the page shows the combined decision for both tasks
with its confidence, **how each of the five branches voted** (green where a
branch agrees with the combined decision, grey where it dissents), and the five
most similar resolved tickets with the resolutions that were applied to them.

Showing the individual votes is the point, not decoration: Table 4.16 lists
"vote margin across base classifiers" as this model's only means of explaining
itself, and the panel can watch that margin collapse on a hard ticket.

### The example to open with

Click **Ambiguous login** ("Cannot login" / "I have been trying since morning").
It is the most useful thirty seconds in the demo:

- classification confidence collapses to **34%** and the branches split 3–2
- routing nonetheless goes to **Help Desk (Tier 1)** at 80% — which is the
  *correct* destination for a ticket carrying no identifying detail
- every retrieved ticket is titled "Cannot login" yet they sit in different
  categories with different resolutions

That is Section 1.2's problem statement — the same words across unrelated
problems — visible on screen, and it is the evidence for the Chapter Five
recommendation that the intake form matters more than a bigger model.

Then click **Wi-Fi in a lecture hall** for the contrast: 100% branch agreement
and Network Team at full confidence.

### Honesty features

These are deliberate, and worth pointing at if the panel asks how the system
behaves when it is wrong:

- **Retrieval confidence.** Top-1 similarity on the held-out set runs 0.44–0.76
  (median 0.63). Above 0.50 retrieval is right 91% of the time; below it, 63%,
  and 12% of queries fall there. Matches under that threshold are coloured
  differently and the page says plainly when nothing in the archive is a close
  precedent, rather than presenting loose matches as good ones.
- **Training/serving parity.** A typed ticket is put through
  `shared.text.build_text` — byte-identical to how the training corpus was
  built, verified on all 4,617 rows. Without this the demo would silently score
  differently-shaped text than the model was fitted on.
- **The held-out set is untouched.** Everything the demo knows was learned from
  the training partition, so the reported metrics still mean what they say.

### Layout

The page is a **fixed frame**: a one-band header, then two columns that scroll
independently and fill the remaining viewport height. This is deliberate for a
presentation — a long result never pushes the input box off screen, the left
column never leaves a growing void beside the results, and the footer disclaimer
stays visible.

Verified at the sizes you are likely to present at:

| Viewport | Result |
|---|---|
| 1920 × 1080 | both decisions, the retrieval warning and four similar tickets all visible at once |
| 1366 × 768 | both decisions fully above the fold; similar tickets scroll in their own pane |
| 1280 × 560 (short) | falls back to ordinary stacked document flow |
| 760 × 900 (narrow) | single column, header wraps, nothing clipped |

The fallback is driven by `@media (max-width:880px),(max-height:600px)` — below
either threshold the fixed frame is dropped rather than squeezed.

### Switching to the evidence dashboard

The server also serves the Chapter Four dashboard at **`/dashboard`**, and both
pages carry a switcher in the top-left, so the whole defence runs from one
browser window. The dashboard is re-read from disk on each request, so
`build_dashboard` picks up without restarting the server. If it has not been
built yet the route says so and gives the command.

### Shareable links

`/?subject=...&description=...` prefills and scores server-side, so a prepared
example arrives with its answer already rendered — no loading flash in front of
an audience.

### Files

| File | Role |
|---|---|
| `export.py` | fits and persists the deployable bundle |
| `predict.py` | inference — importable and testable without a server |
| `serve.py` | the local HTTP server |
| `app.html` | the single-page interface |
