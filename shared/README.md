# Shared layer

What every model draws on, so they differ only in what they model.

| Module | Holds |
|---|---|
| `paths.py` | paths, `SEED`, the CV splitter — import these, never hard-code |
| `data.py` | `load_splits()`, and the two text views |
| `text.py` | cleaning + construction, used by **both** training and serving |
| `encoders.py` | sentence-transformer and TF-IDF→LSA, with the embedding cache |
| `context.py` | context blocks + `unitise`/`assemble` |
| `heads.py` | classifier candidates, tuning, cached selection, ablation |
| `metrics.py` | Table 3.3 metrics, McNemar, bootstrap CIs |
| `artifacts.py` | the inter-model contract — read before changing what a model publishes |

Two traps live here — block scaling and out-of-fold context. Both are in
[NOTES.md](../NOTES.md).
