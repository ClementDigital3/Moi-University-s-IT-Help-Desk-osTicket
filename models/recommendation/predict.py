"""
Inference for model 3 — rank resolved tickets by similarity.

`category` narrows the candidates to what model 1 predicted, which is the
context-aware variant the evaluation selected. Pass None to retrieve over the
whole archive.
"""
import os
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from shared.serving import ticket_frame, encode
from .export import BUNDLE

_B = {}


def load(path=BUNDLE):
    if path not in _B:
        if not os.path.exists(path):
            raise SystemExit(f"no bundle at {path}\nrun:  make m3-export")
        _B[path] = joblib.load(path)
    return _B[path]


def predict(subject="", description="", category=None, top_k=5,
            frame=None, bundle=None):
    b = bundle or load()
    f = frame if frame is not None else ticket_frame(subject, description)
    raw = f"{f['Subject'].iloc[0]}. {f['Description'].iloc[0]}".strip()

    q_sem = encode(b["encoder"], [raw])
    s_sem = (q_sem @ b["embeddings"].T)[0]
    s_lex = cosine_similarity(b["lex_vectorizer"].transform(f["text"]),
                              b["lex_matrix"])[0]
    w = b["blend_w"]
    sim = w * s_sem + (1 - w) * s_lex

    scoped = sim
    filtered = False
    if category:
        mask = np.array(b["categories"]) == category
        if mask.sum() >= top_k:
            scoped = np.where(mask, sim, -np.inf)
            filtered = True

    top = np.argsort(-scoped)[:top_k]
    top = [int(i) for i in top if np.isfinite(scoped[i])]
    recs = b["records"]
    thresh = b["strong_match"]
    out = [{
        "ticket_id": recs[i]["Ticket ID"], "subject": recs[i]["Subject"],
        "category": recs[i]["Category"], "team": recs[i]["Assigned Team"],
        "resolution": recs[i]["Resolution Notes"],
        "similarity": round(float(sim[i]), 4),
        "weak": bool(sim[i] < thresh),
    } for i in top]

    return {
        "matches": out,
        "strong_match": bool(out and out[0]["similarity"] >= thresh),
        "threshold": thresh,
        "top_similarity": out[0]["similarity"] if out else 0.0,
        "category_filtered": filtered,
        "variant": b["best_variant"],
    }
