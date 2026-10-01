"""
Turning a typed ticket into the same thing the models were trained on.

Training works with a DataFrame of thousands of rows; serving has one ticket
and a few fields. This module is the bridge, and it exists so that bridge is
written once rather than three times, slightly differently, in three models.
"""
import os
import numpy as np
import pandas as pd

from .text import build_text

_ENCODERS = {}


def sentence_encoder(name):
    """Load a sentence-transformer once per process and keep it."""
    if name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer
        _ENCODERS[name] = SentenceTransformer(name)
    return _ENCODERS[name]


def encode(name, texts):
    m = sentence_encoder(name)
    return np.asarray(m.encode(list(texts), batch_size=64,
                               show_progress_bar=False, normalize_embeddings=True))


def ticket_frame(subject="", description="", department="", created=None):
    """
    A one-row frame carrying exactly the columns the context blocks read.

    Department and submission time are optional at the prototype's input: the
    ablation found both worth very little, so a ticket without them is scored
    perfectly sensibly. An unknown department one-hots to all zeros, which is
    the honest encoding of 'not stated'.
    """
    return pd.DataFrame([{
        "Subject": subject or "",
        "Description": description or "",
        "Department": department or "",
        "Created": created or pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "text": build_text(subject or "", description or ""),
    }])


def features_for(frame, bundle, extra=None):
    """
    Rebuild this model's exact feature vector for an unseen ticket.

    `bundle` carries the ContextFitter, the selected block keys and the block
    weight that the evaluation settled on, so the vector assembled here is the
    one the reported metrics were measured on.
    """
    from .context import assemble
    E = encode(bundle["encoder"], frame["text"].tolist()
               if bundle.get("encode_clean") else [
                   f"{s}. {d}".strip() for s, d in
                   zip(frame["Subject"].fillna(""), frame["Description"].fillna(""))])
    aux = bundle["fitter"].transform(frame)
    blocks = {"text": (E, E)}
    for k, v in aux.items():
        blocks[k] = (v, v)
    if extra:
        for k, M in extra.items():
            blocks[k] = (bundle["fitter"].apply_extra(k, M),) * 2
    X, _ = assemble(blocks, bundle["keys"], bundle["alpha"])
    return X


def class_scores(model, X, classes):
    """
    Per-class scores for one row, plus what kind of score they are.

    Not every head gives probabilities. Linear SVM gives signed margins, and a
    margin is not a probability -- presenting one as the other would show a
    confident-looking number the model never claimed. So the kind is returned
    alongside, and the interface labels it honestly.

    Margins are softmaxed only to make them comparable across classes; the
    result is a RANKING, which is what the thesis recommends routing be used as.
    """
    order = {c: i for i, c in enumerate(getattr(model, "classes_", classes))}
    widen = lambda v: np.array([v[order[c]] if c in order else -np.inf for c in classes])

    if hasattr(model, "predict_proba"):
        return widen(model.predict_proba(X)[0]), "probability"

    if hasattr(model, "decision_function"):
        d = np.atleast_2d(model.decision_function(X))[0]
        m = widen(d)
        finite = np.isfinite(m)
        e = np.zeros_like(m)
        e[finite] = np.exp(m[finite] - m[finite].max())
        return e / max(e.sum(), 1e-12), "margin"

    s = np.zeros(len(classes))
    s[list(classes).index(model.predict(X)[0])] = 1.0
    return s, "decision"
