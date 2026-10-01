# Model 1 — Ticket classification

Assigns a ticket to one of 7 service categories. Thesis calls this **B1**.
Head of the cascade: models 2 and 3 both consume what it publishes.

```bash
make m1
```

**Input** sentence embedding + department / time / text-shape context blocks
**Output** `results/artifacts/classification.npz` — predictions, probabilities,
and out-of-fold training probabilities (the cascade needs these; see NOTES.md)

## Current result

| | |
|---|---|
| Head | k-NN (cosine), k=15 |
| Features | semantic text + text-shape (weight 0.1) |
| Accuracy / macro F1 | **0.8831** / **0.8793** |
| Out-of-fold agreement | 0.8898 |

The ablation found metadata worth almost nothing once text is read semantically
(department +0.0003, time +0.0002, shape +0.0021). That's a real RQ1 finding: an
implementation can run on ticket text alone.

## Prototype

Its own page at `/classification`, plus the combined `/cascade` view.
Shows the full distribution across all seven categories, not just the label.

```bash
make export && make demo     # then http://127.0.0.1:8000/classification
```

`export.py` persists the head, the fitted `ContextFitter` and the selected
feature set — read from the artifact, so it deploys exactly what was evaluated.
`predict.py` scores one ticket and returns a `proba_vector` that model 2
consumes.

## To extend

- **Feature blocks** — `ABLATION_SPEC` at the top of `run.py`; add the block to
  `shared/context.py` first
- **Classifiers** — `shared/heads.py` (invalidates the head cache)
- **Abstention** — probabilities are already published; thresholding them to send
  low-confidence tickets to a human is the obvious next experiment
