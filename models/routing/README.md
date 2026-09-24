# Model 2 — Resolver routing (context-aware, cascaded)

**Objective Three / RQ3.** Chapter Four calls this model **B2**.

Assigns a ticket to one of the seven resolver teams.

## Architecture

```
subject + description
        │
        ├─ contextual sentence embedding
        ├─ department / temporal / text-shape blocks
        └─ PREDICTED CATEGORY DISTRIBUTION from model 1   ◄── the cascade
        │
        ▼
   classifier head selected independently of model 1
        │
        ▼
   resolver team
```

## The cascade, and the leakage control it needs

This is where "context-aware" stops being a label. Routing is conditioned on
what *kind* of problem the ticket is, not only on its words — and the ablation
shows that conditioning is the single most valuable context block available:

| Feature set | CV macro F1 | Δ |
|---|---|---|
| Semantic text only | 0.7006 | — |
| + department metadata | 0.7031 | +0.0025 |
| + temporal context | 0.6999 | −0.0032 |
| + text-shape context | 0.7025 | +0.0026 |
| **+ predicted-category context** | **0.7098** | **+0.0073** |

The model must never see the *true* category of a training ticket. It consumes
model 1's **out-of-fold** probabilities for training rows and genuine
predictions for held-out rows, so it is fitted against category context of
exactly the quality it meets at inference. Breaking this would inflate the
result and invalidate the comparison — see `shared/artifacts.py`.

## Run

```bash
.venv/bin/python -m models.classification.run    # required upstream
.venv/bin/python -m models.routing.run
```

Fails loudly with the command to run if the upstream artifact is missing.

## Current result

| | |
|---|---|
| Head selected | Linear SVM, C=1.0 |
| Feature set | + predicted-category context (α=0.1) |
| Accuracy | **0.7294** |
| Macro F1 | **0.6888** |
| Correct-routing rate | **0.7294** |

Routing is the hardest of the three decisions and the most expensive to get
wrong — a misrouted ticket burns the wrong team's time before reassignment.
Error concentrates on team pairs described in overlapping language and on the
first-line team, which legitimately receives everything. Chapter Five therefore
recommends deploying this as a **ranked suggestion**, not an assignment.

## Where to extend

- **Top-k routing** — return the top two or three teams with confidence rather
  than an argmax. This is the recommendation the thesis actually makes, and it
  is not yet implemented.
- **Joint formulation** — the cascade inherits model 1's error. A multi-task
  model optimising category and team together is the natural comparison
  (Chapter Five, further research).
- **Reassignment cost** — the corpus carries a `Reassigned` field that no model
  currently uses; a cost-sensitive objective could exploit it.

## Outputs

- `results/artifacts/routing.{npz,json}`
- `results/tables/_m2_{routing,ablation,heads}.csv`
- `results/tables/t4_armB_route_{perclass,confusion}.csv`
