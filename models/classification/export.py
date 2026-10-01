"""
Persist model 1 for serving.

run.py evaluates and throws the fitted objects away. This keeps them, so the
prototype scores a typed ticket with exactly the model the evaluation reported
on -- same head, same feature blocks, same block weight, read from the artifact
rather than re-chosen here.

Run:  make m1-export
Out:  results/deploy/classification.joblib
"""
import os, time
import numpy as np
import joblib

from shared.paths import ROOT, ensure_dirs
from shared.data import load_splits
from shared.encoders import build_embeddings
from shared.context import build_blocks, assemble
from shared.heads import get_head, tune
from shared.artifacts import load_artifact

DEPLOY = os.path.join(ROOT, "results", "deploy")
BUNDLE = os.path.join(DEPLOY, "classification.joblib")

# Which feature-set label maps to which block keys (mirrors run.py's spec)
KEYS_FOR = {
    "Semantic text only": ["text"],
    "+ department metadata": ["text", "dept"],
    "+ temporal context": ["text", "dept", "time"],
    "+ text-shape context": ["text", "dept", "time", "shape"],
}


def main():
    ensure_dirs(); os.makedirs(DEPLOY, exist_ok=True)
    t0 = time.time()
    _, meta = load_artifact("classification")
    head_name, feat, alpha = meta["head"], meta["feature_set"], meta["block_weight"]
    keys = KEYS_FOR[feat]
    print(f"deploying model 1: {head_name} · {feat} · weight {alpha}")

    tr, te, _ = load_splits()
    E_tr, E_te, enc_name, dim, _ = build_embeddings(tr, te)
    blocks, fitter = build_blocks(tr, te, E_tr, E_te)
    X, _ = assemble(blocks, keys, alpha)
    y = tr["Category"].values

    model, cv, params = tune(*get_head(head_name, len(np.unique(y))), X, y)
    print(f"  refitted on {len(tr)} training tickets (CV macro F1 {cv:.4f})")

    joblib.dump({"model": model, "fitter": fitter, "keys": keys, "alpha": alpha,
                 "encoder": enc_name, "classes": sorted(np.unique(y)),
                 "head": head_name, "feature_set": feat,
                 "accuracy": meta.get("accuracy"), "macro_f1": meta.get("macro_f1")},
                BUNDLE, compress=3)
    print(f"  -> {os.path.relpath(BUNDLE, ROOT)} "
          f"({os.path.getsize(BUNDLE)/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
