# Model 1 — Ticket classification (context-aware)

**Objective Three / RQ3.** Also supplies the ablation evidence for Objective One / RQ1.
Chapter Four calls this model **B1**.

Assigns an incoming ticket to one of the seven service categories.

## Architecture

```
subject + description  (raw, uncleaned)
        │
        ├─ contextual sentence embedding        384d, unit norm
        ├─ department metadata block            one-hot, standardised
        ├─ temporal context block               cyclic hour / weekday
        └─ text-shape context block             length, urgency markers
        │
        ▼   each auxiliary block enters at a cross-validated weight α
   classifier head selected from Table 3.2 by 5-fold CV
        │
        ▼
   category  +  class distribution
```

## Why it comes first

It is the head of the cascade. Both other models consume the category
distribution it publishes, so it emits **out-of-fold** training probabilities as
well as held-out ones. Read `shared/artifacts.py` before changing the artifact
shape — the leakage argument depends on it.

## Run

```bash
.venv/bin/python -m models.classification.run          # uses cached embeddings + head scores
.venv/bin/python -m models.classification.run --encoder lsa   # lexical fallback
```

Cold (no cache) this is ~25 min, dominated by head selection. Warm it is ~12 s.
Delete `results/cache/heads_Ticket_classification.json` to force re-selection,
or `results/cache/emb_auto.npz` to re-encode.

## Current result

| | |
|---|---|
| Head selected | k-NN (cosine), k=15 |
| Feature set | semantic text + text-shape context (α=0.1) |
| Accuracy | **0.8831** |
| Macro F1 | **0.8793** |
| Out-of-fold category agreement | **0.8898** |

The ablation found metadata worth very little once the text is read
semantically: department +0.0003, temporal +0.0002, text-shape +0.0021. That is
a real finding for RQ1, not a failure — it means an implementation can be built
on ticket text alone.

## Where to extend

- **Ablation spec** — `ABLATION_SPEC` at the top of `run.py`. Add a block to
  `shared/context.py`, then add a row here.
- **Candidate heads** — `shared/heads.py`. Adding one invalidates the head cache.
- **Abstention** — the class distribution is already published; thresholding it
  to route low-confidence tickets to a human is the natural next experiment
  (Chapter Five recommends it).

## Outputs

- `results/artifacts/classification.{npz,json}` — consumed by models 2 and 3
- `results/tables/_m1_{classification,ablation,heads}.csv` — merged by `analysis/assemble_tables.py`
- `results/tables/t4_armB_cls_{perclass,confusion}.csv`
