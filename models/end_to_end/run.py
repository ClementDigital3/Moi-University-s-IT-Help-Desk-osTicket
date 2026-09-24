"""
THE END-TO-END MODEL, with the individual classifiers as branches.

Design
------
The combined (ensemble) model is the primary artefact. The individual
classifiers of Table 3.2 are not separate experiments that get refitted; they
are BRANCHES of this one fit pass:

        shared TF-IDF (fitted once)
                 |
   +------+------+------+------+------+
   MNB   LogReg  SVM    RF    XGBoost      <- each fitted ONCE, proba cached
   +------+------+------+------+------+
                 |
        combined decision (soft / weighted / majority vote)

Every base model is fitted exactly once per task and its predicted class
probabilities are cached to disk. The combined model is then computed from
those cached probabilities, and so is every individual (branched) result. A
rerun that only changes how the branches are combined costs no refitting at
all -- it reloads the cache.

Efficiency notes (honest, and reported in Chapter 4):
  * The TF-IDF vocabulary is fitted once on the training partition and reused
    for both label tasks instead of being refitted per task.
  * The cheap convex models (MNB, Logistic Regression, Linear SVM) are tuned by
    5-fold grid search, consistent with the hyperparameter-optimisation practice
    cited in Sec. 2.8.
  * Random Forest and XGBoost use fixed, documented settings rather than a grid.
    A single 400-round XGBoost fit cost 33 s on this machine, which put a tuned
    grid beyond a practical runtime; this is a stated compute limitation
    (Sec. 1.9), not a claim that the defaults are optimal.
  * Linear SVM has no native probability output, so it is wrapped in a
    3-fold calibrated classifier to contribute to the soft vote.
"""
import os, json, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier

from shared.paths import TABLES, PREDICTIONS as PRED, CACHE, SEED, ROOT, ensure_dirs
from shared.data import load_splits
from shared.metrics import (clf_metrics, per_class_report, conf_matrix_df,
                            rank_metrics, save_table)

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

TASKS = [("Ticket classification", "Category"),
         ("Resolver routing", "Assigned Team")]


def make_vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.85,
                           sublinear_tf=True, max_features=30000,
                           strip_accents="unicode")


def base_models(n_classes):
    """(name, estimator, grid or None). Grid=None means fixed documented params."""
    return [
        ("Multinomial Naive Bayes",
         MultinomialNB(), {"alpha": [0.05, 0.1, 0.3, 1.0]}),
        ("Logistic Regression",
         LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
         {"C": [1.0, 3.0, 10.0]}),
        ("Linear SVM",
         CalibratedClassifierCV(
             LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000),
             cv=3, method="sigmoid"),
         {"estimator__C": [0.3, 1.0, 3.0]}),
        ("Random Forest",
         RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                                min_samples_leaf=1, random_state=SEED, n_jobs=-1),
         None),
        ("XGBoost",
         XGBClassifier(n_estimators=120, learning_rate=0.25, max_depth=6,
                       subsample=0.9, colsample_bytree=0.7, tree_method="hist",
                       objective="multi:softprob", num_class=n_classes,
                       random_state=SEED, n_jobs=4, verbosity=0),
         None),
    ]


def fit_branch(name, est, grid, Xtr, ytr_i, Xte, classes):
    """Fit one branch once; return (test probabilities, metadata)."""
    t0 = time.time()
    if grid:
        gs = GridSearchCV(est, grid, scoring="f1_macro", cv=CV, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr_i)
        model, cv_score, params = gs.best_estimator_, float(gs.best_score_), gs.best_params_
    else:
        n_jobs = 1 if name == "XGBoost" else -1   # XGB parallelises internally
        cv_score = float(np.mean(cross_val_score(est, Xtr, ytr_i, cv=CV,
                                                 scoring="f1_macro", n_jobs=n_jobs)))
        est.fit(Xtr, ytr_i)
        model, params = est, "fixed (see module docstring)"
    proba = model.predict_proba(Xte)
    # align columns to the global class order
    cols = np.asarray(model.classes_, dtype=int)
    full = np.zeros((proba.shape[0], len(classes)))
    full[:, cols] = proba
    return full, {"cv_f1_macro": round(cv_score, 4),
                  "params": json.dumps(params) if isinstance(params, dict) else params,
                  "fit_s": round(time.time() - t0, 1)}


def combine(probas, weights=None):
    P = np.stack(probas)                       # (n_models, n_samples, n_classes)
    if weights is None:
        return P.mean(axis=0)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.tensordot(w, P, axes=(0, 0))


def majority(probas, n_classes):
    votes = np.stack([p.argmax(axis=1) for p in probas])       # (n_models, n)
    out = np.zeros((votes.shape[1], n_classes))
    for m in range(votes.shape[0]):
        out[np.arange(votes.shape[1]), votes[m]] += 1
    return out


def run_task(task, ycol, tr, te, Xtr, Xte, rows, preds):
    print(f"\n{'='*72}\n{task}  (label: {ycol})\n{'='*72}", flush=True)
    classes = sorted(pd.unique(np.concatenate([tr[ycol].values, te[ycol].values])))
    code = {c: i for i, c in enumerate(classes)}
    inv = {i: c for c, i in code.items()}
    ytr_i = np.array([code[v] for v in tr[ycol].values])
    yte = te[ycol].values

    cache_f = os.path.join(CACHE, f"probas_{ycol.replace(' ', '_')}.npz")
    meta_f = os.path.join(CACHE, f"meta_{ycol.replace(' ', '_')}.json")

    names, probas, metas = [], [], {}
    if os.path.exists(cache_f) and os.path.exists(meta_f):
        z = np.load(cache_f, allow_pickle=True)
        metas = json.load(open(meta_f))
        names = list(z["__names__"])
        probas = [z[n] for n in names]
        print(f"  [cache] reloaded {len(names)} fitted branches -- no refitting")
    else:
        for name, est, grid in base_models(len(classes)):
            p, m = fit_branch(name, est, grid, Xtr, ytr_i, Xte, classes)
            names.append(name); probas.append(p); metas[name] = m
            acc = (np.array([inv[i] for i in p.argmax(1)]) == yte).mean()
            print(f"  fitted {name:26s} cvF1={m['cv_f1_macro']:.4f} "
                  f"testAcc={acc:.4f}  ({m['fit_s']}s)", flush=True)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(cache_f, __names__=np.array(names, dtype=object),
                            **{n: p for n, p in zip(names, probas)})
        json.dump(metas, open(meta_f, "w"), indent=2)

    # ---- branches (individual models, read from the same fit) -------------
    for n, p in zip(names, probas):
        pred = np.array([inv[i] for i in p.argmax(1)])
        row = clf_metrics(yte, pred, task, n)
        row.update({"Arm": "A (branch)", "Role": "individual",
                    "CV F1 (macro)": metas[n]["cv_f1_macro"],
                    "Fit time (s)": metas[n]["fit_s"]})
        rows.append(row)
        preds[f"A|{task}|{n}"] = pred

    # ---- the combined model ----------------------------------------------
    w = [metas[n]["cv_f1_macro"] for n in names]
    variants = {
        "COMBINED (soft vote)":      combine(probas),
        "COMBINED (weighted vote)":  combine(probas, w),
        "COMBINED (majority vote)":  majority(probas, len(classes)),
    }
    for vname, P in variants.items():
        pred = np.array([inv[i] for i in P.argmax(1)])
        row = clf_metrics(yte, pred, task, vname)
        row.update({"Arm": "A (combined)", "Role": "ensemble",
                    "CV F1 (macro)": "", "Fit time (s)": 0.0})
        rows.append(row)
        preds[f"A|{task}|{vname}"] = pred
        print(f"  {vname:30s} acc={row['Accuracy']:.4f}  macroF1={row['F1 (macro)']:.4f}")

    # best overall for this task -> per-class + confusion matrix
    task_rows = [r for r in rows if r["Task"] == task]
    best = max(task_rows, key=lambda r: r["F1 (macro)"])
    tag = "cls" if "classification" in task.lower() else "route"
    bp = preds[f"A|{task}|{best['Model']}"]
    per_class_report(yte, bp, classes).to_csv(
        os.path.join(TABLES, f"t4_armA_{tag}_perclass.csv"), index=False)
    conf_matrix_df(yte, bp, classes).to_csv(
        os.path.join(TABLES, f"t4_armA_{tag}_confusion.csv"))
    print(f"  -> best for this task: {best['Model']} (macro F1 {best['F1 (macro)']:.4f})")


def run_recommendation(tr, te, rows, preds):
    print(f"\n{'='*72}\nResolution recommendation -- TF-IDF cosine retrieval\n{'='*72}")
    corpus = tr[tr["res_clean"].str.strip() != ""].copy().reset_index(drop=True)
    vec = make_vectorizer()
    C = vec.fit_transform(corpus["text"])
    Q = vec.transform(te["text"])
    top = np.argsort(-cosine_similarity(Q, C), axis=1)[:, :10]
    cp = corpus["latent_problem_id"].values
    ranked = [[cp[j] for j in r] for r in top]
    m = rank_metrics(ranked, te["latent_problem_id"].values)
    m.update({"Task": "Resolution recommendation", "Model": "TF-IDF cosine retrieval",
              "Arm": "A (branch)"})
    rows.append(m)
    preds["A|rec|hit1"] = np.array([1 if r[0] == t else 0 for r, t in
                                    zip(ranked, te["latent_problem_id"].values)])
    print(f"  corpus={len(corpus)}  Top-1={m['Top-1']:.4f}  Top-5={m['Top-5']:.4f}  MRR={m['MRR']:.4f}")
    return m


def main():
    ensure_dirs()
    t_start = time.time()
    tr, te, meta = load_splits()

    # shared representation: fitted ONCE, reused by both tasks
    t0 = time.time()
    vec = make_vectorizer()
    Xtr = vec.fit_transform(tr["text"])
    Xte = vec.transform(te["text"])
    print(f"shared TF-IDF fitted once: {Xtr.shape[1]} features "
          f"(train={Xtr.shape[0]}, test={Xte.shape[0]})  [{time.time()-t0:.1f}s]")

    rows, preds = [], {}
    for task, ycol in TASKS:
        run_task(task, ycol, tr, te, Xtr, Xte, rows, preds)
    rec = run_recommendation(tr, te, rows, preds)

    clf_rows = [r for r in rows if r["Task"] != "Resolution recommendation"]
    df = pd.DataFrame(clf_rows).sort_values(["Task", "F1 (macro)"], ascending=[True, False])
    save_table(df, "t43_armA_classification_routing.csv",
               "Table 4.3 -- Arm A: combined model and its branches")
    save_table(pd.DataFrame([rec]), "t45_armA_recommendation.csv",
               "Table 4.5 -- Arm A recommendation")

    np.savez(os.path.join(PRED, "arm_a.npz"),
             **{k: np.asarray(v, dtype=object) for k, v in preds.items()})
    print(f"\ntotal runtime: {time.time()-t_start:.1f}s")
    print(f"branch cache  : {CACHE}  (rerun reuses it; delete to refit)")


if __name__ == "__main__":
    main()
