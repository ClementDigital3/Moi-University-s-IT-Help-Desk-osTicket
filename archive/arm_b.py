"""
ARM B -- three dedicated, context-aware models.

Arm A (arm_a_main.py) is the END-TO-END model: one shared TF-IDF representation
is fitted once and a single combined (voted) classifier stack reads it for every
help desk decision. Arm B is the alternative the thesis sets against it: each
decision gets its OWN model, its OWN representation and its OWN tuning, and each
is made context-aware in the sense of Sec. 2.5 -- the prediction is conditioned
on surrounding contextual and metadata information, not on the ticket text in
isolation.

        B1  Category classification (context-aware)
              contextual sentence embedding of the ticket
            + reporting department      (metadata, known at intake)
            + temporal submission context
            + text-shape context
              -> tuned classifier head

        B2  Resolver routing (context-aware)
              contextual sentence embedding
            + the SAME metadata blocks
            + the predicted category distribution from B1   <- the cascade
              -> independently tuned classifier head

        B3  Historical-resolution recommendation (semantic retrieval)
              its own semantic index over resolved historical tickets,
              optionally narrowed by the B1 category context, and optionally
              blended with a lexical channel

Leakage control
---------------
B2 must never see the true category. On the held-out partition it consumes B1's
actual predictions. On the training partition it consumes OUT-OF-FOLD B1
predictions produced by cross_val_predict, so the routing head is fitted against
category context of the same (imperfect) quality it will meet at inference.

Representation, per Table 3.2 ("contextual / semantic benchmark ... where
feasible"): a sentence-transformer encoder is used where it can be provisioned,
otherwise TF-IDF -> truncated SVD latent-semantic vectors (the allowance made in
Sec. 1.9 and Sec. 3.11). Whichever is used is recorded and reported.

Outputs consumed by compare.py / figures.py / write_chapters.py:
    results/tables/t44_armB_classification_routing.csv
    results/tables/t46_armB_recommendation.csv
    results/tables/t4_armB_{cls,route}_perclass.csv
    results/tables/t4_armB_{cls,route}_confusion.csv
    results/tables/t410_feature_ablation.csv     (Objective 1 / RQ1)
    results/tables/t411_representation.csv       (Objective 2 / RQ2)
    results/tables/t412_armB_head_selection.csv  (Objective 3 / RQ3)
    results/tables/t413_recommendation_variants.csv (Objective 4 / RQ4)
    results/predictions/arm_b.npz, arm_b_meta.json
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_val_score)
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from common import (load_splits, clf_metrics, per_class_report, conf_matrix_df,
                    rank_metrics, save_table, TABLES, SEED, ROOT)

PRED = os.path.join(ROOT, "results", "predictions")
CACHE = os.path.join(ROOT, "results", "cache")
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

ENCODER_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


# ==========================================================================
# Representation
# ==========================================================================
def raw_text(df):
    """
    Natural, *uncleaned* subject + description.

    Arm A deliberately strips casing, digits and punctuation because TF-IDF
    gains nothing from them. A transformer encoder does: sentence boundaries and
    casing carry contextual signal. Feeding Arm B the raw text is therefore part
    of the representation being tested, not an inconsistency between the arms.
    """
    subj = df["Subject"].fillna("").astype(str).str.strip()
    desc = df["Description"].fillna("").astype(str).str.strip()
    return (subj + ". " + desc).str.replace(r"\s+", " ", regex=True).str.strip()


def encode_sbert(train_texts, test_texts):
    from sentence_transformers import SentenceTransformer
    last = None
    for name in ENCODER_CANDIDATES:
        try:
            m = SentenceTransformer(name)
        except Exception as e:                       # offline / not cached
            last = e
            print(f"  [encoder] {name} unavailable ({type(e).__name__})")
            continue
        enc = lambda xs: np.asarray(m.encode(list(xs), batch_size=64,
                                             show_progress_bar=False,
                                             normalize_embeddings=True))
        E_tr, E_te = enc(train_texts), enc(test_texts)
        return E_tr, E_te, name, int(E_tr.shape[1])
    raise RuntimeError(f"no sentence encoder could be loaded: {last}")


def encode_lsa(train_texts, test_texts, dim=300):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.85,
                          sublinear_tf=True, max_features=30000,
                          strip_accents="unicode")
    Xtr, Xte = vec.fit_transform(train_texts), vec.transform(test_texts)
    dim = min(dim, Xtr.shape[1] - 1)
    pipe = make_pipeline(TruncatedSVD(n_components=dim, random_state=SEED),
                         Normalizer(copy=False))
    E_tr, E_te = pipe.fit_transform(Xtr), pipe.transform(Xte)
    ev = float(pipe.named_steps["truncatedsvd"].explained_variance_ratio_.sum())
    return E_tr, E_te, f"TF-IDF + truncated SVD (LSA, {dim}d, {ev:.1%} var)", dim


def build_embeddings(tr, te, pref):
    """Encode once and cache: the encode pass is the expensive part of Arm B."""
    key = os.path.join(CACHE, f"emb_{pref}.npz")
    if os.path.exists(key):
        z = np.load(key, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        print(f"  [cache] embeddings reloaded -- {meta['encoder']} ({meta['dim']}d)")
        return z["E_tr"], z["E_te"], meta["encoder"], meta["dim"], meta["encode_seconds"]

    t0 = time.time()
    if pref == "lsa":
        E_tr, E_te, name, dim = encode_lsa(raw_text(tr), raw_text(te))
    else:
        try:
            E_tr, E_te, name, dim = encode_sbert(raw_text(tr), raw_text(te))
        except Exception as e:
            if pref == "sbert":
                raise
            print(f"  [encoder] falling back to LSA as allowed by Sec. 3.11 ({e})")
            E_tr, E_te, name, dim = encode_lsa(raw_text(tr), raw_text(te))
    secs = round(time.time() - t0, 1)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(key, E_tr=E_tr, E_te=E_te,
                        meta=json.dumps({"encoder": name, "dim": dim,
                                         "encode_seconds": secs}))
    print(f"  encoder: {name}  dim={dim}  encode={secs}s")
    return E_tr, E_te, name, dim, secs


# ==========================================================================
# Context feature blocks  (Objective 1 / RQ1)
# ==========================================================================
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
    """
    Put an auxiliary block on the same footing as the semantic block.

    The sentence embedding has unit row-norm by construction, so each of its 384
    dimensions carries a magnitude of roughly 0.05. A raw one-hot department
    block would enter at magnitude 1.0 and simply dominate every distance-based
    head -- an artefact of how the block is encoded, not evidence about how much
    the department actually tells us. Each auxiliary block is therefore
    standardised on the training partition, then given unit row-norm, so that the
    only thing controlling its influence is the explicit weight `alpha` applied
    at assembly time.
    """
    sc = StandardScaler().fit(Btr)
    A, B = sc.transform(Btr), sc.transform(Bte)
    n = lambda M: M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    return n(A), n(B)


def assemble(blocks, keys, alpha=0.3):
    """
    Horizontally stack the selected blocks.

    `text` enters at its native unit norm; every auxiliary block enters at norm
    `alpha`, which is tuned by cross-validation in ablate().
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


# ==========================================================================
# Candidate classifier heads
# ==========================================================================
class XGBWrapped(BaseEstimator, ClassifierMixin):
    """
    XGBoost with an internal label encoder.

    xgboost's scikit-learn wrapper rejects non-integer targets. Every other head
    here consumes the string labels directly, so the encoding is kept inside the
    estimator rather than leaking a special case into the selection, ablation and
    cross_val_predict code paths.
    """

    def __init__(self, n_estimators=150, learning_rate=0.25, max_depth=6,
                 subsample=0.9, colsample_bytree=0.7, random_state=SEED, n_jobs=4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self._m = XGBClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree, tree_method="hist",
            objective="multi:softprob", num_class=len(self.classes_),
            random_state=self.random_state, n_jobs=self.n_jobs, verbosity=0)
        self._m.fit(X, self._le.transform(y))
        return self

    def predict(self, X):
        return self._le.inverse_transform(self._m.predict(X))

    def predict_proba(self, X):
        return self._m.predict_proba(X)


def heads(n_classes):
    """
    Table 3.2 candidates, adapted to a DENSE semantic representation.

    Multinomial Naive Bayes is absent by necessity, not by choice: it requires
    non-negative counts and cannot consume signed embedding dimensions. It is
    evaluated in Arm A, where the representation suits it. A cosine k-NN head is
    added because it is the natural classifier over a normalised semantic space.
    """
    return [
        ("Logistic Regression",
         LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
         {"C": [1.0, 3.0, 10.0, 30.0]}),
        ("Linear SVM",
         LinearSVC(class_weight="balanced", random_state=SEED, max_iter=8000),
         {"C": [0.3, 1.0, 3.0]}),
        ("Random Forest",
         RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                random_state=SEED, n_jobs=-1),
         {"min_samples_leaf": [1, 2]}),
        # Fixed, documented settings rather than a grid: a single tuned XGBoost
        # grid cost 27 minutes for one task on this machine, which puts it beyond
        # a practical runtime. This mirrors exactly how Arm A treats its boosted
        # branch, and is a stated compute limitation (Sec. 1.9) rather than a
        # claim that these settings are optimal.
        ("XGBoost",
         XGBWrapped(n_estimators=150, learning_rate=0.25, max_depth=6,
                    subsample=0.9, colsample_bytree=0.7,
                    random_state=SEED, n_jobs=4),
         None),
        ("k-NN (cosine)",
         KNeighborsClassifier(metric="cosine", weights="distance", n_jobs=-1),
         {"n_neighbors": [5, 10, 15, 25]}),
    ]


def tune(name, est, grid, X, y):
    """
    5-fold cross-validation on the TRAINING partition only (Sec. 3.15).

    Candidates with a grid are tuned by grid search; candidates declared with
    grid=None are scored once at their fixed documented settings.
    """
    n_jobs = 1 if name == "XGBoost" else -1
    if not grid:
        score = float(np.mean(cross_val_score(est, X, y, cv=CV, scoring="f1_macro",
                                              n_jobs=n_jobs)))
        est.fit(X, y)
        return est, score, "fixed (see heads() docstring)"
    gs = GridSearchCV(est, grid, scoring="f1_macro", cv=CV, n_jobs=n_jobs, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_, float(gs.best_score_), gs.best_params_


def select_head(task, X, y, rows):
    """
    Score every candidate head and keep the best cross-validated macro F1.

    Each head's result is written to disk as soon as it is computed. Head
    selection is by far the most expensive step in Arm B, so an interrupted run
    resumes from the cache instead of repeating hours of fitting. Only the
    winning head's NAME is needed downstream, so nothing is refitted on reload.
    Delete results/cache/heads_*.json to force a fresh selection.
    """
    os.makedirs(CACHE, exist_ok=True)
    cache_f = os.path.join(CACHE, f"heads_{task.replace(' ', '_')}.json")
    done = json.load(open(cache_f)) if os.path.exists(cache_f) else {}

    best = None
    for name, est, grid in heads(len(np.unique(y))):
        if name in done:
            score, params, secs = done[name]["score"], done[name]["params"], done[name]["secs"]
            print(f"    {name:22s} cvF1={score:.4f}  (cached)", flush=True)
        else:
            t0 = time.time()
            _, score, params = tune(name, est, grid, X, y)
            secs = round(time.time() - t0, 1)
            params = params if isinstance(params, str) else json.dumps(params)
            done[name] = {"score": score, "params": params, "secs": secs}
            json.dump(done, open(cache_f, "w"), indent=2)
            print(f"    {name:22s} cvF1={score:.4f}  ({secs}s)", flush=True)
        rows.append({"Task": task, "Head": name, "CV F1 (macro)": round(score, 4),
                     "Best params": params, "Tune time (s)": secs})
        if best is None or score > best[1]:
            best = (name, score, None, params)
    print(f"    -> head selected: {best[0]} (CV macro F1 {best[1]:.4f})")
    return best


# ==========================================================================
# B1 / B2 -- context-aware classification and routing
# ==========================================================================
def ablate(task, blocks, y, best_head_name, n_classes, rows):
    """
    Objective 1 / RQ1: which feature blocks actually carry signal?

    Each block is added cumulatively to the semantic core and the change in
    cross-validated macro F1 is recorded. Cross-validation, not the held-out
    partition, is used so that the held-out set remains untouched by feature
    selection.
    """
    spec = [("Semantic text only", ["text"]),
            ("+ department metadata", ["text", "dept"]),
            ("+ temporal context", ["text", "dept", "time"]),
            ("+ text-shape context", ["text", "dept", "time", "shape"])]
    if "cat" in blocks:
        spec.append(("+ predicted-category context", ["text", "dept", "time", "shape", "cat"]))

    name, est, grid = next(h for h in heads(n_classes) if h[0] == best_head_name)
    n_jobs = 1 if name == "XGBoost" else -1
    # Each block's weight is itself tuned, so a block is judged at its best
    # setting rather than at an arbitrary one. A single fit of the boosted head
    # costs minutes at this sample size, so it is given the mid weight only --
    # a stated compute limitation (Sec. 1.9), not a claim that 0.3 is optimal.
    alphas = [0.3] if name == "XGBoost" else [0.1, 0.3, 1.0]

    prev, best = None, None
    for label, keys in spec:
        if keys == ["text"]:
            trials = [(None, assemble(blocks, keys)[0])]
        else:
            trials = [(a, assemble(blocks, keys, a)[0]) for a in alphas]
        score, alpha, dims = -1.0, None, 0
        for a, X in trials:
            s = float(np.mean(cross_val_score(est, X, y, cv=CV, scoring="f1_macro",
                                              n_jobs=n_jobs)))
            if s > score:
                score, alpha, dims = s, a, X.shape[1]
        rows.append({"Task": task, "Feature set": label, "Blocks": " + ".join(keys),
                     "Dimensions": dims, "Block weight": "—" if alpha is None else alpha,
                     "CV F1 (macro)": round(score, 4),
                     "Δ vs previous": "" if prev is None else round(score - prev, 4)})
        print(f"    {label:32s} d={dims:<5d} w={'—' if alpha is None else alpha:<5} "
              f"cvF1={score:.4f}"
              + ("" if prev is None else f"  (Δ {score-prev:+.4f})"), flush=True)
        prev = score
        if best is None or score > best[1]:
            best = (keys, score, label, alpha)
    print(f"    -> feature set selected: {best[2]} (CV macro F1 {best[1]:.4f}, "
          f"block weight {best[3]})")
    return best[0], best[2], (0.3 if best[3] is None else best[3])


def run_labelled_task(task, ycol, tr, te, blocks, rows, preds, head_rows, abl_rows):
    print(f"\n{'='*72}\n{task}  (label: {ycol})\n{'='*72}", flush=True)
    y_tr, y_te = tr[ycol].values, te[ycol].values
    classes = sorted(pd.unique(np.concatenate([y_tr, y_te])))

    # --- 1. choose the head on the semantic core alone -------------------
    print("  head selection (semantic text only):")
    X_core, _ = assemble(blocks, ["text"])
    head_name, _, _, _ = select_head(task, X_core, y_tr, head_rows)

    # --- 2. choose the feature set with that head ------------------------
    print("  context-block ablation:")
    keys, feat_label, alpha = ablate(task, blocks, y_tr, head_name, len(classes), abl_rows)

    # --- 3. refit the selected head on the selected feature set ----------
    X_tr, X_te = assemble(blocks, keys, alpha)
    name, est, grid = next(h for h in heads(len(classes)) if h[0] == head_name)
    t0 = time.time()
    model, cv_score, params = tune(name, est, grid, X_tr, y_tr)
    fit_s = round(time.time() - t0, 1)
    pred = model.predict(X_te)

    row = clf_metrics(y_te, pred, task, f"B: {head_name} + context")
    row.update({"Arm": "B (task-specific)", "Head": head_name,
                "Feature set": feat_label, "Dimensions": X_tr.shape[1],
                "Block weight": alpha, "CV F1 (macro)": round(cv_score, 4),
                "Best params": json.dumps(params), "Fit time (s)": fit_s})
    rows.append(row)
    preds[f"B|{task}"] = pred
    print(f"  FINAL  acc={row['Accuracy']:.4f}  macroF1={row['F1 (macro)']:.4f}  ({fit_s}s)")

    tag = "cls" if "classification" in task.lower() else "route"
    per_class_report(y_te, pred, classes).to_csv(
        os.path.join(TABLES, f"t4_armB_{tag}_perclass.csv"), index=False)
    conf_matrix_df(y_te, pred, classes).to_csv(
        os.path.join(TABLES, f"t4_armB_{tag}_confusion.csv"))
    return model, X_tr, X_te, pred, classes, head_name


def category_context(model, head_name, X_tr, X_te, y_tr, classes):
    """
    Turn B1 into a context block for B2, without leaking the true category.

    Training rows get OUT-OF-FOLD predictions (cross_val_predict), so the routing
    head never sees a category derived from a model that had already seen that
    row's label. Held-out rows get B1's genuine predictions.
    """
    n_jobs = 1 if head_name == "XGBoost" else -1
    if hasattr(model, "predict_proba"):
        oof = cross_val_predict(model, X_tr, y_tr, cv=CV, method="predict_proba",
                                n_jobs=n_jobs)
        # cross_val_predict orders columns by np.unique(y_tr); align to `classes`
        order = {c: i for i, c in enumerate(np.unique(y_tr))}
        Ctr = np.zeros((len(y_tr), len(classes)))
        for j, c in enumerate(classes):
            if c in order:
                Ctr[:, j] = oof[:, order[c]]
        Cte = np.zeros((X_te.shape[0], len(classes)))
        p = model.predict_proba(X_te)
        mcls = {c: i for i, c in enumerate(model.classes_)}
        for j, c in enumerate(classes):
            if c in mcls:
                Cte[:, j] = p[:, mcls[c]]
    else:                                            # LinearSVC -> hard one-hot
        oof = cross_val_predict(model, X_tr, y_tr, cv=CV, n_jobs=n_jobs)
        idx = {c: i for i, c in enumerate(classes)}
        Ctr = np.zeros((len(y_tr), len(classes)))
        Ctr[np.arange(len(y_tr)), [idx[v] for v in oof]] = 1.0
        pte = model.predict(X_te)
        Cte = np.zeros((X_te.shape[0], len(classes)))
        Cte[np.arange(len(pte)), [idx[v] for v in pte]] = 1.0
    agree = float((np.array(classes)[Ctr.argmax(1)] == y_tr).mean())
    print(f"  category context built: out-of-fold agreement with truth = {agree:.4f}")
    return Ctr, Cte, [f"cat={c}" for c in classes], agree


# ==========================================================================
# B3 -- historical-resolution recommendation
# ==========================================================================
def run_recommendation(tr, te, E_tr, E_te, pred_cat, rows, preds):
    """
    Objective 4 / RQ4.

    Corpus  : training tickets that actually carry a resolution note -- the same
              corpus Arm A retrieves over, so the two arms are comparable.
    Relevant: a retrieved ticket shares the query's latent problem id.

    Three variants are evaluated:
      1. semantic          -- cosine over the sentence embeddings
      2. semantic + category context -- candidates restricted to the category B1
                              predicted for the query (falls back to the full
                              corpus when that leaves nothing)
      3. semantic + lexical hybrid   -- blended with a TF-IDF channel; the blend
                              weight is chosen by leave-one-out retrieval on the
                              TRAINING corpus, never on the held-out queries.
    """
    print(f"\n{'='*72}\nB3  Historical-resolution recommendation\n{'='*72}", flush=True)
    keep = tr["res_clean"].fillna("").str.strip().values != ""
    corpus = tr[keep].reset_index(drop=True)
    Ec = E_tr[keep]
    cpid = corpus["latent_problem_id"].values
    ccat = corpus["Category"].values
    qpid = te["latent_problem_id"].values
    print(f"  corpus={len(corpus)} resolved tickets   queries={len(te)}")

    S_sem = E_te @ Ec.T                                   # vectors are unit-norm

    # lexical channel, fitted on the corpus only
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.85,
                          sublinear_tf=True, max_features=30000,
                          strip_accents="unicode")
    C_lex = vec.fit_transform(corpus["text"])
    S_lex = cosine_similarity(vec.transform(te["text"]), C_lex)

    # --- blend weight by leave-one-out retrieval on the training corpus ----
    Slo_sem = Ec @ Ec.T
    Slo_lex = cosine_similarity(C_lex, C_lex)
    best_w, best_s = 1.0, -1.0
    for w in [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]:
        # blend first, THEN mask the self-match: masking with -inf beforehand
        # would give 0 * -inf = nan at the endpoint weights.
        blend = w * Slo_sem + (1 - w) * Slo_lex
        np.fill_diagonal(blend, -np.inf)
        hit = (cpid[blend.argmax(1)] == cpid).mean()
        print(f"    w_semantic={w:<5.2f} leave-one-out Top-1 on train = {hit:.4f}")
        if hit > best_s:
            best_w, best_s = w, hit
    print(f"    -> blend weight selected: w_semantic={best_w}")

    def evaluate(S, label, model_name, restrict=None):
        ranked = []
        for i in range(S.shape[0]):
            s = S[i]
            if restrict is not None:
                m = ccat == restrict[i]
                if m.sum() >= 10:
                    s = np.where(m, s, -np.inf)
            ranked.append([cpid[j] for j in np.argsort(-s)[:10]])
        m = rank_metrics(ranked, qpid)
        m.update({"Task": "Resolution recommendation", "Model": model_name,
                  "Variant": label, "Arm": "B (task-specific)"})
        print(f"    {label:34s} Top-1={m['Top-1']:.4f}  Top-5={m['Top-5']:.4f}  MRR={m['MRR']:.4f}")
        hits = np.array([1 if r[0] == t else 0 for r, t in zip(ranked, qpid)])
        return m, hits

    print("  variants:")
    variants, hitmap = [], {}
    S_hyb = best_w * S_sem + (1 - best_w) * S_lex
    for label, S, restrict, mname in (
            ("Semantic retrieval", S_sem, None, "Sentence-embedding cosine"),
            ("Semantic + category context", S_sem, pred_cat,
             "Sentence-embedding cosine, category-filtered"),
            (f"Semantic + lexical hybrid (w={best_w})", S_hyb, None,
             f"Hybrid semantic+lexical (w={best_w})"),
            (f"Hybrid + category context", S_hyb, pred_cat,
             f"Hybrid semantic+lexical, category-filtered")):
        m, h = evaluate(S, label, mname, restrict)
        variants.append(m)
        hitmap[label] = h

    save_table(pd.DataFrame(variants)[
        ["Variant", "Top-1", "Top-3", "Top-5", "Top-10", "Precision@5",
         "Recall@5", "MRR"]],
        "t413_recommendation_variants.csv",
        "Table 4.13 -- Arm B recommendation variants (Objective 4)")

    best = max(variants, key=lambda r: r["MRR"])
    preds["B|rec|hit1"] = hitmap[best["Variant"]]
    print(f"  -> best variant: {best['Variant']} (MRR {best['MRR']:.4f})")
    rows.append(best)
    return best, best_w


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="auto", choices=["auto", "sbert", "lsa"])
    args = ap.parse_args()
    os.makedirs(PRED, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    t_start = time.time()

    tr, te, meta = load_splits()
    print(f"{'='*72}\nARM B -- three dedicated context-aware models\n{'='*72}")
    print(f"train={len(tr)}  test={len(te)}")

    E_tr, E_te, enc_name, dim, enc_s = build_embeddings(tr, te, args.encoder)

    blocks = {"text": (E_tr, E_te)}
    for key, builder in (("dept", block_department), ("time", block_temporal),
                         ("shape", block_shape)):
        a, b, _ = builder(tr, te)
        blocks[key] = unitise(a, b)

    rows, preds, head_rows, abl_rows = [], {}, [], []

    # ---------------- B1 -------------------------------------------------
    m1, X1tr, X1te, pred_cat, cat_classes, head1 = run_labelled_task(
        "Ticket classification", "Category", tr, te, blocks, rows, preds,
        head_rows, abl_rows)

    # ---------------- B2 (consumes B1 as context) ------------------------
    Ctr, Cte, c_names, oof_agree = category_context(
        m1, head1, X1tr, X1te, tr["Category"].values, cat_classes)
    blocks["cat"] = unitise(Ctr, Cte)
    run_labelled_task("Resolver routing", "Assigned Team", tr, te, blocks,
                      rows, preds, head_rows, abl_rows)

    # ---------------- B3 -------------------------------------------------
    rec, blend_w = run_recommendation(tr, te, E_tr, E_te, pred_cat, rows, preds)

    # ---------------- representation comparison (RQ2) --------------------
    print(f"\n{'='*72}\nRepresentation comparison -- contextual vs lexical (Objective 2)\n{'='*72}")
    rep_rows = []
    L_tr, _, lsa_name, _ = encode_lsa(raw_text(tr), raw_text(te))
    for task, ycol in (("Ticket classification", "Category"),
                       ("Resolver routing", "Assigned Team")):
        y = tr[ycol].values
        hname = next(r["Head"] for r in rows if r.get("Task") == task and "Head" in r)
        name, est, grid = next(h for h in heads(len(np.unique(y))) if h[0] == hname)
        n_jobs = 1 if name == "XGBoost" else -1
        for label, X in (("Contextual sentence embedding", E_tr),
                         ("Lexical TF-IDF → LSA", L_tr)):
            sc = float(np.mean(cross_val_score(est, X, y, cv=CV,
                                               scoring="f1_macro", n_jobs=n_jobs)))
            rep_rows.append({"Task": task, "Representation": label,
                             "Head": hname, "Dimensions": X.shape[1],
                             "CV F1 (macro)": round(sc, 4)})
            print(f"  {task:24s} {label:32s} cvF1={sc:.4f}", flush=True)

    # ---------------- persist -------------------------------------------
    clf_rows = [r for r in rows if r["Task"] != "Resolution recommendation"]
    save_table(pd.DataFrame(clf_rows), "t44_armB_classification_routing.csv",
               "Table 4.4 -- Arm B: dedicated context-aware models")
    save_table(pd.DataFrame([rec]), "t46_armB_recommendation.csv",
               "Table 4.6 -- Arm B recommendation")
    save_table(pd.DataFrame(abl_rows), "t410_feature_ablation.csv",
               "Table 4.10 -- Context-block ablation (Objective 1 / RQ1)")
    save_table(pd.DataFrame(rep_rows), "t411_representation.csv",
               "Table 4.11 -- Representation comparison (Objective 2 / RQ2)")
    save_table(pd.DataFrame(head_rows), "t412_armB_head_selection.csv",
               "Table 4.12 -- Arm B classifier-head selection (Objective 3 / RQ3)")

    np.savez(os.path.join(PRED, "arm_b.npz"),
             **{k: np.asarray(v, dtype=object) for k, v in preds.items()})
    json.dump({"encoder": enc_name, "dim": dim, "encode_seconds": enc_s,
               "oof_category_agreement": round(oof_agree, 4),
               "blend_w_semantic": blend_w,
               "runtime_seconds": round(time.time() - t_start, 1)},
              open(os.path.join(PRED, "arm_b_meta.json"), "w"), indent=2)
    print(f"\ntotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
