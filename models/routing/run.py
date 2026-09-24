"""
MODEL 2 -- Resolver routing, context-aware and cascaded.

Objective Three / RQ3. Referred to as B2 in Chapter Four.

    ticket subject + description
        -> contextual sentence embedding
        +  department / temporal / text-shape context blocks
        +  PREDICTED CATEGORY from model 1          <- the cascade
        -> independently selected and tuned classifier head
        -> resolver team

The cascade is what makes this model context-aware in the strong sense: routing
is conditioned on what KIND of problem the ticket is, not only on its words. It
consumes model 1's published artifact and never the true category -- training
rows get out-of-fold category context, held-out rows get genuine predictions.

Correct-routing rate (Table 3.3) is the proportion of tickets reaching the right
resolver without reassignment; on a single-assignment task that is the accuracy.

Run:  .venv/bin/python -m models.routing.run      (requires model 1 to have run)
Outputs:
    results/artifacts/routing.npz
    results/tables/_m2_*.csv, t4_armB_route_{perclass,confusion}.csv
"""
import os, json, time, argparse
import numpy as np
import pandas as pd

from shared.paths import TABLES, ensure_dirs
from shared.data import load_splits
from shared.encoders import build_embeddings
from shared.context import build_blocks, assemble, unitise
from shared.heads import select_head, ablate, get_head, tune
from shared.metrics import clf_metrics, per_class_report, conf_matrix_df
from shared.artifacts import save_artifact, load_artifact

TASK = "Resolver routing"
YCOL = "Assigned Team"

ABLATION_SPEC = [
    ("Semantic text only", ["text"]),
    ("+ department metadata", ["text", "dept"]),
    ("+ temporal context", ["text", "dept", "time"]),
    ("+ text-shape context", ["text", "dept", "time", "shape"]),
    ("+ predicted-category context", ["text", "dept", "time", "shape", "cat"]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="auto", choices=["auto", "sbert", "lsa"])
    args = ap.parse_args()
    ensure_dirs()
    t_start = time.time()

    tr, te, meta = load_splits()
    print(f"{'='*72}\nMODEL 2 -- {TASK} (context-aware, cascaded on model 1)\n{'='*72}")
    print(f"train={len(tr)}  test={len(te)}")

    E_tr, E_te, enc_name, dim, enc_s = build_embeddings(tr, te, args.encoder)
    blocks = build_blocks(tr, te, E_tr, E_te)

    # ---- the cascade: model 1's category distribution becomes a context block
    up, up_meta = load_artifact("classification")
    print(f"  upstream: model 1 ({up_meta.get('head','?')}), "
          f"out-of-fold category agreement {up_meta.get('oof_category_agreement','?')}")
    blocks["cat"] = unitise(up["oof_proba"].astype(float),
                            up["test_proba"].astype(float))

    y_tr, y_te = tr[YCOL].values, te[YCOL].values
    classes = sorted(pd.unique(np.concatenate([y_tr, y_te])))

    head_rows, abl_rows = [], []
    print("\n  head selection (semantic text only):")
    X_core, _ = assemble(blocks, ["text"])
    head_name = select_head(TASK, X_core, y_tr, head_rows)

    print("\n  context-block ablation (the last row is the cascade):")
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
    print(f"  FINAL  acc={row['Accuracy']:.4f}  macroF1={row['F1 (macro)']:.4f}  "
          f"correct-routing={row['Correct-routing rate']:.4f}  ({fit_s}s)")

    # did the cascade actually earn its place?
    d = pd.DataFrame(abl_rows)
    if "+ predicted-category context" in set(d["Feature set"]):
        casc = float(d[d["Feature set"] == "+ predicted-category context"]["Δ vs previous"].iloc[0])
        print(f"  cascade contribution: {casc:+.4f} macro F1 over the same model "
              f"without category context")

    per_class_report(y_te, pred, classes).to_csv(
        os.path.join(TABLES, "t4_armB_route_perclass.csv"), index=False)
    conf_matrix_df(y_te, pred, classes).to_csv(
        os.path.join(TABLES, "t4_armB_route_confusion.csv"))
    pd.DataFrame([row]).to_csv(os.path.join(TABLES, "_m2_routing.csv"), index=False)
    pd.DataFrame(abl_rows).to_csv(os.path.join(TABLES, "_m2_ablation.csv"), index=False)
    pd.DataFrame(head_rows).to_csv(os.path.join(TABLES, "_m2_heads.csv"), index=False)

    save_artifact("routing",
                  meta={"head": head_name, "feature_set": feat_label,
                        "block_weight": alpha, "encoder": enc_name,
                        "upstream": "classification",
                        "accuracy": row["Accuracy"], "macro_f1": row["F1 (macro)"],
                        "correct_routing_rate": row["Correct-routing rate"],
                        "runtime_seconds": round(time.time() - t_start, 1)},
                  test_pred=pred, classes=np.array(classes, dtype=object))
    print(f"\ntotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
