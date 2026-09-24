# Shared layer

Everything the models have in common, so they differ only in what they actually
model — not in how they load a split or compute an F1.

| Module | Holds |
|---|---|
| `paths.py` | canonical paths, `SEED`, the 5-fold `CV` splitter. Import these; never hard-code a path. |
| `data.py` | `load_splits()`, and the two text views: `clean_text` (lexical) and `raw_text` (contextual) |
| `encoders.py` | sentence-transformer and TF-IDF→LSA representations, with the embedding cache |
| `context.py` | the context blocks — department, temporal, text-shape — plus `unitise`/`assemble` |
| `heads.py` | Table 3.2 classifier candidates, tuning, cached head selection, the ablation loop |
| `metrics.py` | Table 3.3 metrics, confusion matrices, McNemar, bootstrap CIs |
| `artifacts.py` | **the inter-model contract** — read this before changing what a model publishes |

## Two things that are easy to get wrong

**Block scaling.** The semantic block has unit row-norm, so each of its 384
dimensions carries magnitude ≈0.05. A raw one-hot department block enters at
magnitude 1.0 and swamps every distance-based head — which during development
cost routing 0.28 macro F1 and looked like a finding about departments. It was
an artefact of encoding width. `unitise()` standardises and renormalises every
auxiliary block, and `assemble()` applies an explicit weight that `ablate()`
cross-validates. Do not bypass them.

**Out-of-fold context.** Any model cascaded on another must consume out-of-fold
predictions for training rows, never predictions from a model that already saw
those rows' labels. `artifacts.py` explains why and the classification model
implements it. Breaking this inflates results silently.
