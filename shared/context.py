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


def _dept_matrix(df, cats):
    """Reporting department as a one-hot over a FIXED vocabulary."""
    idx = {c: i for i, c in enumerate(cats)}
    M = np.zeros((len(df), len(cats)))
    for r, v in enumerate(df["Department"].fillna("").astype(str).values):
        if v in idx:
            M[r, idx[v]] = 1.0
    return M


def _time_matrix(df):
    """Submission time encoded cyclically, plus a working-hours indicator."""
    t = pd.to_datetime(df["Created"], errors="coerce")
    hour = t.dt.hour.fillna(12).values.astype(float)
    dow = t.dt.dayofweek.fillna(2).values.astype(float)
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        (dow >= 5).astype(float),
        ((hour >= 8) & (hour < 17)).astype(float),
    ])


def _shape_matrix(df):
    """Text-shape context: how much the user actually wrote, and how."""
    s = raw_text(df)
    subj = df["Subject"].fillna("").astype(str)
    return np.column_stack([
        np.log1p(s.str.split().str.len().values.astype(float)),
        np.log1p(s.str.len().values.astype(float)),
        np.log1p(subj.str.split().str.len().values.astype(float)),
        s.str.contains(r"\?").values.astype(float),
        s.str.contains(r"\d").values.astype(float),
        s.str.contains(r"(?i)urgent|asap|immediately").values.astype(float),
    ])


BLOCK_NAMES = {
    "dept": lambda df, st: _dept_matrix(df, st["dept_cats"]),
    "time": lambda df, st: _time_matrix(df),
    "shape": lambda df, st: _shape_matrix(df),
}


def _unit_rows(M):
    return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)


class ContextFitter:
    """
    Builds the auxiliary context blocks, and REMEMBERS how.

    Training could get away with fitting and discarding: every row it will ever
    see is present at fit time. Serving cannot -- a ticket typed into the
    prototype has to be standardised against the same statistics the model was
    fitted on, and one-hot encoded against the same department vocabulary.
    Refitting a scaler on a single row would produce nonsense.

    So this object is what a model persists at export time, and it is the only
    definition of how a block is built. Both training and serving go through it,
    which is what stops the two drifting apart.
    """

    def __init__(self):
        self.state = {}
        self.scalers = {}

    def fit(self, tr):
        self.state["dept_cats"] = sorted(tr["Department"].fillna("").astype(str).unique())
        for key, build in BLOCK_NAMES.items():
            self.scalers[key] = StandardScaler().fit(build(tr, self.state))
        return self

    def transform(self, df):
        """The unitised auxiliary blocks for any frame, seen or unseen."""
        return {key: _unit_rows(self.scalers[key].transform(build(df, self.state)))
                for key, build in BLOCK_NAMES.items()}

    def fit_extra(self, key, M):
        """
        Register a block this fitter does not know how to build itself.

        Resolver routing's category context is the obvious case: it comes from
        another model, so only its SCALING belongs here. Fitted on the
        out-of-fold training matrix, applied unchanged at inference.
        """
        self.scalers[key] = StandardScaler().fit(M)
        return _unit_rows(self.scalers[key].transform(M))

    def apply_extra(self, key, M):
        return _unit_rows(self.scalers[key].transform(M))


def build_blocks(tr, te, E_tr, E_te):
    """
    The standard block set every labelled model starts from.

    Returns ({"text","dept","time","shape"}, fitter). Keep the fitter: it is
    what export.py persists so the prototype can rebuild these blocks for a
    ticket nobody has seen.
    """
    fitter = ContextFitter().fit(tr)
    btr, bte = fitter.transform(tr), fitter.transform(te)
    blocks = {"text": (E_tr, E_te)}
    for key in BLOCK_NAMES:
        blocks[key] = (btr[key], bte[key])
    return blocks, fitter


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
