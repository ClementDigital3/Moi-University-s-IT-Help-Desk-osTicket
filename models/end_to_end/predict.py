"""
Inference for the end-to-end model.

Kept separate from serve.py so the prediction logic can be tested and reused
without starting a web server, and so the demo cannot quietly diverge from what
the evaluation measured.

Every ticket goes through shared.text.build_text -- the same construction the
training corpus went through -- before it reaches the vectorizer.
"""
import os
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from shared.text import build_text
from .export import BUNDLE

_CACHE = {}

# Calibrated on the held-out set: top-1 retrieval similarity there runs
# 0.44-0.76 (median 0.63). Above 0.50 retrieval is right 91% of the time;
# below it, 63%, and 12% of queries fall there. A query scoring under this is
# usually one whose wording the archive simply has no close precedent for, so
# the prototype says so rather than presenting loose matches as if they were
# good ones.
STRONG_MATCH = 0.50


def load(path=BUNDLE):
    """Load the deployed bundle once and keep it in memory."""
    if path not in _CACHE:
        if not os.path.exists(path):
            raise SystemExit(
                f"no deployed bundle at {path}\n"
                f"export one first:\n"
                f"    .venv/bin/python -m models.end_to_end.export")
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]


def _branch_proba(est, X, n_classes):
    """One branch's probabilities, widened to the task's full class order."""
    p = est.predict_proba(X)
    full = np.zeros((p.shape[0], n_classes))
    full[:, np.asarray(est.classes_, dtype=int)] = p
    return full


def _combine(probas, rule, weights, n_classes):
    P = np.stack(probas)                       # (branches, rows, classes)
    if rule == "majority":
        votes = np.zeros((P.shape[1], n_classes))
        for m in range(P.shape[0]):
            votes[np.arange(P.shape[1]), P[m].argmax(1)] += 1
        return votes / P.shape[0]
    if rule == "weighted":
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        return np.tensordot(w, P, axes=(0, 0))
    return P.mean(axis=0)


def predict(subject, description, top_k=5, bundle=None):
    """
    Score one ticket.

    Returns the combined decision for each task, every individual branch's vote
    (so the demo can show the ensemble actually working), and the most similar
    resolved tickets with the resolutions that were applied to them.
    """
    b = bundle or load()
    text = build_text(subject or "", description or "")
    if not text.strip():
        return {"error": "Enter a subject or a description."}

    X = b["vectorizer"].transform([text])
    out = {"cleaned_text": text, "tasks": {}}

    for task, spec in b["tasks"].items():
        classes, n = spec["classes"], len(spec["classes"])
        probas, branches = [], []
        for name, est in spec["branches"]:
            p = _branch_proba(est, X, n)[0]
            probas.append(p[None, :])
            j = int(p.argmax())
            branches.append({"model": name, "label": classes[j],
                             "confidence": round(float(p[j]), 4)})
        combined = _combine(probas, spec["rule"], spec["weights"], n)[0]
        order = np.argsort(-combined)
        out["tasks"][task] = {
            "rule": spec["rule"],
            "label": classes[int(order[0])],
            "confidence": round(float(combined[order[0]]), 4),
            "runner_up": classes[int(order[1])],
            "runner_up_confidence": round(float(combined[order[1]]), 4),
            "distribution": [{"label": classes[int(i)],
                              "score": round(float(combined[i]), 4)} for i in order],
            "branches": branches,
            "agreement": round(
                sum(1 for x in branches if x["label"] == classes[int(order[0])])
                / len(branches), 2),
        }

    r = b["retrieval"]
    sim = cosine_similarity(r["vectorizer"].transform([text]), r["matrix"])[0]
    top = np.argsort(-sim)[:top_k]
    strong = bool(len(top) and sim[int(top[0])] >= STRONG_MATCH)
    out["retrieval"] = {
        "strong_match": strong,
        "threshold": STRONG_MATCH,
        "top_similarity": round(float(sim[int(top[0])]), 4) if len(top) else 0.0,
    }
    out["similar"] = [{
        "ticket_id": r["records"][int(i)]["Ticket ID"],
        "subject": r["records"][int(i)]["Subject"],
        "category": r["records"][int(i)]["Category"],
        "team": r["records"][int(i)]["Assigned Team"],
        "resolution": r["records"][int(i)]["Resolution Notes"],
        "similarity": round(float(sim[int(i)]), 4),
        "weak": bool(sim[int(i)] < STRONG_MATCH),
    } for i in top if sim[int(i)] > 0]
    return out


if __name__ == "__main__":
    import sys, json
    subj = sys.argv[1] if len(sys.argv) > 1 else "Cannot access elearning portal"
    desc = sys.argv[2] if len(sys.argv) > 2 else "I keep getting invalid credentials."
    print(json.dumps(predict(subj, desc), indent=2)[:2000])
