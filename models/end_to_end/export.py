"""
Export the end-to-end model as a deployable bundle.

run.py exists to produce the thesis tables: it fits, evaluates, and throws the
fitted objects away. A prototype needs the opposite -- the fitted objects kept,
so a ticket typed into the demo is scored in milliseconds by exactly the models
the evaluation reported on.

What gets written to results/deploy/end_to_end.joblib:

    vectorizer      the shared TF-IDF, fitted on the training partition
    branches        the five Table 3.2 classifiers per task, fitted
    weights         each branch's CV macro F1, for the weighted vote
    rule            the combination rule that won each task in Table 4.5
    retrieval       the resolution-recommendation index and its corpus records

Fitted on the TRAINING partition only -- never on the held-out set -- so the
demo is honest: everything it shows you was learned from data the reported
metrics were not computed on.

Run:  .venv/bin/python -m models.end_to_end.export
"""
import os, json, time
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from shared.paths import ROOT, TABLES, SEED, ensure_dirs
from shared.data import load_splits
from shared.encoders import tfidf_vectorizer

DEPLOY = os.path.join(ROOT, "results", "deploy")
BUNDLE = os.path.join(DEPLOY, "end_to_end.joblib")

TASKS = [("Ticket classification", "Category"),
         ("Resolver routing", "Assigned Team")]


def base_models(n_classes):
    """The same five branches run.py evaluates, at the same settings."""
    return [
        ("Multinomial Naive Bayes", MultinomialNB(alpha=0.1)),
        ("Logistic Regression",
         LogisticRegression(max_iter=2000, class_weight="balanced", C=3.0,
                            random_state=SEED)),
        ("Linear SVM",
         CalibratedClassifierCV(
             LinearSVC(class_weight="balanced", C=1.0, random_state=SEED,
                       max_iter=5000), cv=3, method="sigmoid")),
        ("Random Forest",
         RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample",
                                min_samples_leaf=1, random_state=SEED, n_jobs=-1)),
        ("XGBoost",
         XGBClassifier(n_estimators=120, learning_rate=0.25, max_depth=6,
                       subsample=0.9, colsample_bytree=0.7, tree_method="hist",
                       objective="multi:softprob", num_class=n_classes,
                       random_state=SEED, n_jobs=4, verbosity=0)),
    ]


def winning_rule(task):
    """
    Which combination rule Table 4.5 selected for this task.

    Read from the results rather than hard-coded, so the demo deploys whatever
    the evaluation actually favoured -- including the case where combining lost
    to a single branch, which is what happened on resolver routing.
    """
    p = os.path.join(TABLES, "t43_armA_classification_routing.csv")
    if not os.path.exists(p):
        return "soft"
    df = pd.read_csv(p)
    d = df[(df["Task"] == task) & (df["Arm"] == "A (combined)")]
    if d.empty:
        return "soft"
    best = d.loc[d["F1 (macro)"].idxmax(), "Model"]
    for key in ("soft", "weighted", "majority"):
        if key in best:
            return key
    return "soft"


def branch_cv_weights(ycol):
    """Each branch's cross-validated macro F1, cached by run.py."""
    p = os.path.join(ROOT, "results", "cache", f"meta_{ycol.replace(' ', '_')}.json")
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return {k: v["cv_f1_macro"] for k, v in m.items()}


def main():
    ensure_dirs()
    os.makedirs(DEPLOY, exist_ok=True)
    t0 = time.time()

    tr, te, meta = load_splits()
    print(f"{'='*72}\nExporting the end-to-end model for deployment\n{'='*72}")
    print(f"fitting on the TRAINING partition only: {len(tr)} tickets")

    vec = tfidf_vectorizer()
    Xtr = vec.fit_transform(tr["text"])
    print(f"shared TF-IDF: {Xtr.shape[1]} features")

    bundle = {"vectorizer": vec, "tasks": {}, "meta": {
        "n_train": len(tr), "exported": time.strftime("%Y-%m-%d %H:%M"),
        "note": "fitted on the training partition only; held-out set untouched"}}

    for task, ycol in TASKS:
        classes = sorted(tr[ycol].unique())
        code = {c: i for i, c in enumerate(classes)}
        y = np.array([code[v] for v in tr[ycol].values])
        rule = winning_rule(task)
        w = branch_cv_weights(ycol)
        print(f"\n{task}  ({len(classes)} classes, rule: {rule} vote)")

        fitted = []
        for name, est in base_models(len(classes)):
            t = time.time()
            est.fit(Xtr, y)
            fitted.append((name, est))
            print(f"  fitted {name:26s} ({time.time()-t:.1f}s)", flush=True)

        bundle["tasks"][task] = {
            "label_column": ycol, "classes": classes, "rule": rule,
            "branches": fitted,
            "weights": [w.get(n, 1.0) if w else 1.0 for n, _ in fitted],
        }

    # ---- retrieval index -------------------------------------------------
    corpus = tr[tr["res_clean"].str.strip() != ""].reset_index(drop=True)
    rvec = tfidf_vectorizer()
    C = rvec.fit_transform(corpus["text"])
    bundle["retrieval"] = {
        "vectorizer": rvec, "matrix": C,
        "records": corpus[["Ticket ID", "Subject", "Category", "Assigned Team",
                           "Resolution Notes"]].to_dict("records"),
    }
    print(f"\nretrieval index: {C.shape[0]} resolved tickets, {C.shape[1]} features")

    joblib.dump(bundle, BUNDLE, compress=3)
    size = os.path.getsize(BUNDLE) / 1e6
    print(f"\nbundle -> {os.path.relpath(BUNDLE, ROOT)}  ({size:.1f} MB)")
    print(f"total: {time.time()-t0:.1f}s")
    print(f"\nnext: .venv/bin/python -m models.end_to_end.serve")


if __name__ == "__main__":
    main()
