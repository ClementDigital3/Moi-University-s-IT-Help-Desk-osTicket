"""
Context feature blocks -- the "context-aware" part of the models (Objective 1 / RQ1).

A ticket is more than its words: it arrives from a department, at a time, with a
certain shape, and (once predicted) in a category. Each of those is built here
as a separate BLOCK so that its contribution can be measured independently
rather than assumed.

Admissibility. Only fields available at the moment a ticket arrives may be used.
Resolution notes, the resolved timestamp, subcategory and assigned resolver are
excluded: the first two are post-hoc, and the last two are themselves triage
outcomes rather than inputs to triage.

Scaling. The semantic block has unit row-norm by construction, so each of its
dimensions carries a magnitude of roughly 1/sqrt(d). A raw one-hot department
block would enter at magnitude 1.0 and dominate every distance-based head --
an artefact of encoding width, not evidence about the department. Every
auxiliary block is therefore standardised and renormalised by unitise(), and
enters at an explicit weight `alpha` that is itself cross-validated.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import raw_text


def block_department(tr, te):
    """Reporting department -- structured metadata, known the moment a ticket opens."""
    cats = sorted(tr["Department"].fillna("").unique())
    idx = {c: i for i, c in enumerate(cats)}

    def oh(df):
        M = np.zeros((len(df), len(cats)))
        for r, v in enumerate(df["Department"].fillna("").values):
            if v in idx:
                M[r, idx[v]] = 1.0
        return M
    return oh(tr), oh(te), [f"dept={c}" for c in cats]


def block_temporal(tr, te):
    """Submission time encoded cyclically, plus a working-hours indicator."""
    def feats(df):
        t = pd.to_datetime(df["Created"], errors="coerce")
        hour = t.dt.hour.fillna(12).values.astype(float)
        dow = t.dt.dayofweek.fillna(2).values.astype(float)
        return np.column_stack([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            ((dow >= 5).astype(float)),
            (((hour >= 8) & (hour < 17)).astype(float)),
        ])
    names = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "in_hours"]
    return feats(tr), feats(te), names


def block_shape(tr, te):
    """Text-shape context: how much the user actually wrote, and how."""
    def feats(df):
        s = raw_text(df)
        subj = df["Subject"].fillna("").astype(str)
        n_tok = s.str.split().str.len().values.astype(float)
        return np.column_stack([
            np.log1p(n_tok),
            np.log1p(s.str.len().values.astype(float)),
            np.log1p(subj.str.split().str.len().values.astype(float)),
            s.str.contains(r"\?").values.astype(float),
            s.str.contains(r"\d").values.astype(float),
            s.str.contains(r"(?i)urgent|asap|immediately").values.astype(float),
        ])
    names = ["log_tokens", "log_chars", "log_subject_tokens",
             "has_question", "has_digit", "has_urgency"]
    return feats(tr), feats(te), names


def unitise(Btr, Bte):
    """Standardise on the training partition, then give every row unit norm."""
    sc = StandardScaler().fit(Btr)
    A, B = sc.transform(Btr), sc.transform(Bte)
    n = lambda M: M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    return n(A), n(B)


def build_blocks(tr, te, E_tr, E_te):
    """
    The standard block set every labelled model starts from.

    Returns {"text", "dept", "time", "shape"}. A model that has upstream context
    to add -- resolver routing adds the predicted category -- inserts its own
    key afterwards, already passed through unitise().
    """
    blocks = {"text": (E_tr, E_te)}
    for key, builder in (("dept", block_department), ("time", block_temporal),
                         ("shape", block_shape)):
        a, b, _ = builder(tr, te)
        blocks[key] = unitise(a, b)
    return blocks


def assemble(blocks, keys, alpha=0.3):
    """
    Horizontally stack the selected blocks.

    `text` enters at its native unit norm; every auxiliary block enters at norm
    `alpha`, which is cross-validated in heads.ablate().
    """
    tr_parts, te_parts = [], []
    for k in keys:
        a, b = blocks[k]
        if k == "text":
            tr_parts.append(a)
            te_parts.append(b)
        else:
            tr_parts.append(alpha * a)
            te_parts.append(alpha * b)
    return np.hstack(tr_parts), np.hstack(te_parts)
