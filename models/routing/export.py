"""
Persist Arm B model B2 for serving.

Deploys exactly what the evaluation settled on, read from the artifact. The
category block's scaler is fitted on B1's OUT-OF-FOLD training matrix and
saved with the bundle, so at inference a live category prediction is scaled the
same way the training ones were.

Run:  make m2-export
Out:  results/deploy/routing.joblib
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
BUNDLE = os.path.join(DEPLOY, "routing.joblib")

KEYS_FOR = {
    "Semantic text only": ["text"],
    "+ department metadata": ["text", "dept"],
    "+ temporal context": ["text", "dept", "time"],
    "+ text-shape context": ["text", "dept", "time", "shape"],
    "+ predicted-category context": ["text", "dept", "time", "shape", "cat"],
}


def main():
    ensure_dirs(); os.makedirs(DEPLOY, exist_ok=True)
    t0 = time.time()
    _, meta = load_artifact("routing")
    head_name, feat, alpha = meta["head"], meta["feature_set"], meta["block_weight"]
    keys = KEYS_FOR[feat]
    print(f"deploying B2: {head_name} · {feat} · weight {alpha}")

    up, up_meta = load_artifact("classification")
    tr, te, _ = load_splits()
    E_tr, E_te, enc_name, dim, _ = build_embeddings(tr, te)
    blocks, fitter = build_blocks(tr, te, E_tr, E_te)
    # scaler fitted on out-of-fold category context, exactly as in run.py
    blocks["cat"] = (fitter.fit_extra("cat", up["oof_proba"].astype(float)),
                     fitter.apply_extra("cat", up["test_proba"].astype(float)))
    print(f"  category context from B1 (out-of-fold agreement "
          f"{up_meta.get('oof_category_agreement')})")

    X, _ = assemble(blocks, keys, alpha)
    y = tr["Assigned Team"].values
    model, cv, params = tune(*get_head(head_name, len(np.unique(y))), X, y)
    print(f"  refitted on {len(tr)} training tickets (CV macro F1 {cv:.4f})")

    joblib.dump({"model": model, "fitter": fitter, "keys": keys, "alpha": alpha,
                 "encoder": enc_name, "classes": sorted(np.unique(y)),
                 "category_classes": [str(c) for c in up["classes"]],
                 "head": head_name, "feature_set": feat,
                 "needs_category": "cat" in keys,
                 "accuracy": meta.get("accuracy"), "macro_f1": meta.get("macro_f1"),
                 "correct_routing_rate": meta.get("correct_routing_rate")},
                BUNDLE, compress=3)
    print(f"  -> {os.path.relpath(BUNDLE, ROOT)} "
          f"({os.path.getsize(BUNDLE)/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
