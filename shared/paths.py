"""Canonical paths and experiment constants. Import these; never hard-code a path."""
import os
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROC = os.path.join(ROOT, "data", "processed")

RESULTS = os.path.join(ROOT, "results")
TABLES = os.path.join(RESULTS, "tables")
FIGURES = os.path.join(RESULTS, "figures")
PREDICTIONS = os.path.join(RESULTS, "predictions")
ARTIFACTS = os.path.join(RESULTS, "artifacts")     # inter-model handoff
CACHE = os.path.join(RESULTS, "cache")             # expensive intermediates

DOCS = os.path.join(ROOT, "docs")

# Fixed across every model so that all comparisons are paired and reproducible.
SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def ensure_dirs():
    for d in (TABLES, FIGURES, PREDICTIONS, ARTIFACTS, CACHE, DOCS):
        os.makedirs(d, exist_ok=True)
