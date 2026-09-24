"""
MODEL 1 -- Ticket classification, context-aware.

Objective Three / RQ3 (and it supplies the evidence for Objective One / RQ1).
Referred to as B1 in Chapter Four.

    ticket subject + description
        -> contextual sentence embedding
        +  department / temporal / text-shape context blocks
        -> independently selected and tuned classifier head
        -> service category

This is the head of the cascade. Both downstream models consume the category
distribution it publishes, so it also emits OUT-OF-FOLD training probabilities
(see shared/artifacts.py for why).

Run:  .venv/bin/python -m models.classification.run
Outputs:
    results/artifacts/classification.npz       (consumed by routing + recommendation)
    results/tables/t44_armB_classification_routing.csv   (this model's row)
    results/tables/t410_feature_ablation.csv             (this model's rows)
    results/tables/t412_armB_head_selection.csv          (this model's rows)
    results/tables/t4_armB_cls_{perclass,confusion}.csv
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

from shared.paths import TABLES, CV, ensure_dirs
from shared.data import load_splits
from shared.encoders import build_embeddings
from shared.context import build_blocks, assemble
from shared.heads import select_head, ablate, get_head, tune
from shared.metrics import clf_metrics, per_class_report, conf_matrix_df
from shared.artifacts import save_artifact

TASK = "Ticket classification"
YCOL = "Category"

ABLATION_SPEC = [
    ("Semantic text only", ["text"]),
    ("+ department metadata", ["text", "dept"]),
    ("+ temporal context", ["text", "dept", "time"]),
    ("+ text-shape context", ["text", "dept", "time", "shape"]),
]


def out_of_fold_proba(model, head_name, X_tr, y_tr, classes):
    """
    Category context for the downstream models, produced without leakage.

    Each training row's distribution comes from folds that excluded it, so a
    model fitted on this meets context of the same quality it meets at
    inference. LinearSVC has no predict_proba, so it contributes a hard one-hot.
    """
    n_jobs = 1 if head_name == "XGBoost" else -1
    if hasattr(model, "predict_proba"):
        oof = cross_val_predict(model, X_tr, y_tr, cv=CV, method="predict_proba",
                                n_jobs=n_jobs)
        order = {c: i for i, c in enumerate(np.unique(y_tr))}
        P = np.zeros((len(y_tr), len(classes)))
        for j, c in enumerate(classes):
            if c in order:
                P[:, j] = oof[:, order[c]]
    else:
        oof = cross_val_predict(model, X_tr, y_tr, cv=CV, n_jobs=n_jobs)
        idx = {c: i for i, c in enumerate(classes)}
        P = np.zeros((len(y_tr), len(classes)))
        P[np.arange(len(y_tr)), [idx[v] for v in oof]] = 1.0
    return P


def test_proba(model, X_te, classes):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_te)
        mcls = {c: i for i, c in enumerate(model.classes_)}
        P = np.zeros((X_te.shape[0], len(classes)))
        for j, c in enumerate(classes):
            if c in mcls:
                P[:, j] = p[:, mcls[c]]
        return P
    pred = model.predict(X_te)
    idx = {c: i for i, c in enumerate(classes)}
    P = np.zeros((len(pred), len(classes)))
    P[np.arange(len(pred)), [idx[v] for v in pred]] = 1.0
    return P


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="auto", choices=["auto", "sbert", "lsa"])
    args = ap.parse_args()
    ensure_dirs()
    t_start = time.time()

    tr, te, meta = load_splits()
    print(f"{'='*72}\nMODEL 1 -- {TASK} (context-aware)\n{'='*72}")
    print(f"train={len(tr)}  test={len(te)}")

    E_tr, E_te, enc_name, dim, enc_s = build_embeddings(tr, te, args.encoder)
    blocks = build_blocks(tr, te, E_tr, E_te)

    y_tr, y_te = tr[YCOL].values, te[YCOL].values
    classes = sorted(pd.unique(np.concatenate([y_tr, y_te])))

    head_rows, abl_rows = [], []
    print("\n  head selection (semantic text only):")
    X_core, _ = assemble(blocks, ["text"])
    head_name = select_head(TASK, X_core, y_tr, head_rows)

    print("\n  context-block ablation:")
    keys, feat_label, alpha = ablate(TASK, blocks, y_tr, head_name, len(classes),
                                     abl_rows, ABLATION_SPEC)

    print("\n  final fit:")
    X_tr, X_te = assemble(blocks, keys, alpha)
    name, est, grid = get_head(head_name, len(classes))
    t0 = time.time()
    model, cv_score, params = tune(name, est, grid, X_tr, y_tr)
    fit_s = round(time.time() - t0, 1)
    pred = model.predict(X_te)

    row = clf_metrics(y_te, pred, TASK, f"B: {head_name} + context")
    row.update({"Arm": "B (task-specific)", "Head": head_name,
                "Feature set": feat_label, "Dimensions": X_tr.shape[1],
                "Block weight": alpha, "CV F1 (macro)": round(cv_score, 4),
                "Best params": params if isinstance(params, str) else json.dumps(params),
                "Fit time (s)": fit_s})
    print(f"  FINAL  acc={row['Accuracy']:.4f}  macroF1={row['F1 (macro)']:.4f}  ({fit_s}s)")

    per_class_report(y_te, pred, classes).to_csv(
        os.path.join(TABLES, "t4_armB_cls_perclass.csv"), index=False)
    conf_matrix_df(y_te, pred, classes).to_csv(
        os.path.join(TABLES, "t4_armB_cls_confusion.csv"))
    pd.DataFrame([row]).to_csv(os.path.join(TABLES, "_m1_classification.csv"), index=False)
    pd.DataFrame(abl_rows).to_csv(os.path.join(TABLES, "_m1_ablation.csv"), index=False)
    pd.DataFrame(head_rows).to_csv(os.path.join(TABLES, "_m1_heads.csv"), index=False)

    print("\n  publishing category context for the downstream models:")
    P_oof = out_of_fold_proba(model, head_name, X_tr, y_tr, classes)
    agree = float((np.array(classes)[P_oof.argmax(1)] == y_tr).mean())
    print(f"    out-of-fold agreement with truth = {agree:.4f}")

    save_artifact("classification",
                  meta={"head": head_name, "feature_set": feat_label,
                        "block_weight": alpha, "encoder": enc_name, "dim": dim,
                        "encode_seconds": enc_s,
                        "oof_category_agreement": round(agree, 4),
                        "accuracy": row["Accuracy"], "macro_f1": row["F1 (macro)"],
                        "runtime_seconds": round(time.time() - t_start, 1)},
                  test_pred=pred, test_proba=test_proba(model, X_te, classes),
                  oof_proba=P_oof, classes=np.array(classes, dtype=object))
    print(f"\ntotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
