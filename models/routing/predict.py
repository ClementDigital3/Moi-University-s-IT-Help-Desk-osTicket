"""
Inference for model 2 — resolver team, conditioned on model 1's category.

`category` is model 1's probability vector. Pass it in rather than recomputing:
the cascade is the architecture, and making the dependency explicit keeps it
visible instead of hidden inside this function.
"""
import os
import numpy as np
import joblib

from shared.serving import ticket_frame, features_for, class_scores
from .export import BUNDLE

_B = {}


def load(path=BUNDLE):
    if path not in _B:
        if not os.path.exists(path):
            raise SystemExit(f"no bundle at {path}\nrun:  make m2-export")
        _B[path] = joblib.load(path)
    return _B[path]


def predict(subject="", description="", department="", category=None,
            frame=None, bundle=None):
    b = bundle or load()
    f = frame if frame is not None else ticket_frame(subject, description, department)
    extra = None
    if b["needs_category"]:
        if category is None:
            raise ValueError("this model is cascaded: pass model 1's proba_vector "
                             "as `category`")
        extra = {"cat": np.asarray(category, dtype=float).reshape(1, -1)}
    X = features_for(f, b, extra=extra)

    classes = list(b["classes"])
    probs, kind = class_scores(b["model"], X, classes)
    o = np.argsort(-probs)
    return {
        "label": classes[int(o[0])],
        "confidence": round(float(probs[o[0]]), 4),
        "runner_up": classes[int(o[1])],
        "runner_up_confidence": round(float(probs[o[1]]), 4),
        "ranked": [{"label": classes[int(i)], "score": round(float(probs[i]), 4)}
                   for i in o],
        "model": f"{b['head']} · {b['feature_set']}",
        "score_type": kind,
        "cascaded": bool(b["needs_category"]),
    }


def card():
    b = load()
    return {
        "name": "Resolver routing",
        "thesis": "B2",
        "objective": "Objective 3 / RQ3",
        "head": b["head"],
        "features": b["feature_set"],
        "encoder": b["encoder"],
        "classes": len(b["classes"]),
        "metrics": [
            {"label": "Accuracy", "value": b.get("accuracy")},
            {"label": "Macro F1", "value": b.get("macro_f1")},
            {"label": "Correct-routing rate", "value": b.get("correct_routing_rate")},
        ],
        "note": ("Cascaded on classification: conditioned on the predicted category, "
                 "which the ablation found worth +0.0073 macro F1 — more than every "
                 "other context block combined."),
        "cascaded": bool(b["needs_category"]),
    }
