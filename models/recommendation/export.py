"""
Persist model 3 for serving.

Saves the semantic index over resolved tickets, the lexical channel, the blend
weight the evaluation chose, and the records themselves so the prototype can
show what was actually done about each match.

Run:  make m3-export
Out:  results/deploy/recommendation.joblib
"""
import os, time
import joblib

from shared.paths import ROOT, ensure_dirs
from shared.data import load_splits
from shared.encoders import build_embeddings, tfidf_vectorizer
from shared.artifacts import load_artifact

DEPLOY = os.path.join(ROOT, "results", "deploy")
BUNDLE = os.path.join(DEPLOY, "recommendation.joblib")

# Calibrated on the held-out set, as in the end-to-end prototype: below this,
# the archive has no close precedent and the interface says so.
STRONG_MATCH = 0.50


def main():
    ensure_dirs(); os.makedirs(DEPLOY, exist_ok=True)
    t0 = time.time()
    _, meta = load_artifact("recommendation")
    w = meta["blend_w_semantic"]
    print(f"deploying model 3: {meta['best_variant']} · blend w_semantic={w}")

    tr, te, _ = load_splits()
    E_tr, _, enc_name, dim, _ = build_embeddings(tr, te)
    keep = tr["res_clean"].fillna("").str.strip().values != ""
    corpus = tr[keep].reset_index(drop=True)

    vec = tfidf_vectorizer()
    C_lex = vec.fit_transform(corpus["text"])
    print(f"  index: {len(corpus)} resolved tickets")

    joblib.dump({
        "embeddings": E_tr[keep], "lex_vectorizer": vec, "lex_matrix": C_lex,
        "encoder": enc_name, "blend_w": w, "strong_match": STRONG_MATCH,
        "categories": corpus["Category"].tolist(),
        "records": corpus[["Ticket ID", "Subject", "Category", "Assigned Team",
                           "Resolution Notes"]].to_dict("records"),
        "best_variant": meta["best_variant"], "top1": meta.get("top1"),
        "mrr": meta.get("mrr"),
    }, BUNDLE, compress=3)
    print(f"  -> {os.path.relpath(BUNDLE, ROOT)} "
          f"({os.path.getsize(BUNDLE)/1e6:.1f} MB, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
