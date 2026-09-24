"""
Candidate classifier heads and the selection machinery shared by the labelled
models (ticket classification and resolver routing).

The candidates are those of Table 3.2, adapted to a dense semantic
representation. Selection is always by five-fold cross-validation on the
TRAINING partition only (Sec. 3.15); the held-out partition is touched once, at
the very end, by the model that owns it.
"""
import os, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from .paths import CACHE, SEED, CV
from .context import assemble


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
    evaluated in the end-to-end model, where the representation suits it. A
    cosine k-NN head is added because it is the natural classifier over a
    normalised semantic space.
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
        # a practical runtime. This mirrors how the end-to-end model treats its
        # boosted branch, and is a stated compute limitation (Sec. 1.9) rather
        # than a claim that these settings are optimal.
        ("XGBoost",
         XGBWrapped(n_estimators=150, learning_rate=0.25, max_depth=6,
                    subsample=0.9, colsample_bytree=0.7,
                    random_state=SEED, n_jobs=4),
         None),
        ("k-NN (cosine)",
         KNeighborsClassifier(metric="cosine", weights="distance", n_jobs=-1),
         {"n_neighbors": [5, 10, 15, 25]}),
    ]


def get_head(name, n_classes):
    """Look one candidate up by name, unfitted."""
    return next(h for h in heads(n_classes) if h[0] == name)


def tune(name, est, grid, X, y):
    """
    Five-fold cross-validation on the TRAINING partition only (Sec. 3.15).

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
    selection is by far the most expensive step, so an interrupted run resumes
    from the cache instead of repeating hours of fitting. Only the winning
    head's NAME is needed downstream, so nothing is refitted on reload.
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
            best = (name, score, params)
    print(f"    -> head selected: {best[0]} (CV macro F1 {best[1]:.4f})")
    return best[0]


def ablate(task, blocks, y, head_name, n_classes, rows, spec):
    """
    Objective 1 / RQ1: measure what each context block is actually worth.

    `spec` is a list of (label, block_keys) added cumulatively. Each block's
    weight is itself tuned, so a block is judged at its best setting rather than
    penalised for how it happens to be encoded. Cross-validation, not the
    held-out partition, is used so that feature selection never touches the test
    data.

    Returns (selected_keys, selected_label, selected_alpha).
    """
    name, est, grid = get_head(head_name, n_classes)
    n_jobs = 1 if name == "XGBoost" else -1
    # A single fit of the boosted head costs minutes at this sample size, so it
    # is given the mid weight only -- a stated compute limitation (Sec. 1.9).
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
