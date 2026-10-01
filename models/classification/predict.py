"""Inference for model 1 — category from a typed ticket."""
import os
import numpy as np
import joblib

from shared.serving import ticket_frame, features_for, class_scores
from .export import BUNDLE

_B = {}


def load(path=BUNDLE):
    if path not in _B:
        if not os.path.exists(path):
            raise SystemExit(f"no bundle at {path}\nrun:  make m1-export")
        _B[path] = joblib.load(path)
    return _B[path]


def predict(subject="", description="", department="", frame=None, bundle=None):
    b = bundle or load()
    f = frame if frame is not None else ticket_frame(subject, description, department)
    X = features_for(f, b)
    classes = list(b["classes"])
    probs, kind = class_scores(b["model"], X, classes)
    o = np.argsort(-probs)
    return {
        "label": classes[int(o[0])],
        "confidence": round(float(probs[o[0]]), 4),
        "runner_up": classes[int(o[1])],
        "runner_up_confidence": round(float(probs[o[1]]), 4),
        "distribution": [{"label": classes[int(i)], "score": round(float(probs[i]), 4)}
                         for i in o],
        "proba_vector": probs.reshape(1, -1),   # what model 2 consumes
        "classes": classes,
        "model": f"{b['head']} · {b['feature_set']}",
        "score_type": kind,
    }


if __name__ == "__main__":
    import sys, json
    r = predict(sys.argv[1] if len(sys.argv) > 1 else "Cannot upload course materials",
                sys.argv[2] if len(sys.argv) > 2 else "The elearning platform rejects my notes.")
    r.pop("proba_vector")
    print(json.dumps(r, indent=2))
