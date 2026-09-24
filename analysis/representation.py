"""
Objective Two / RQ2 -- contextual versus lexical representation.

Section 3.11 requires that the superiority of contextual representations be
tested rather than assumed. This is that test, and it is deliberately NOT part
of any one model: it is a cross-cutting comparison that holds the classifier and
the cross-validation folds fixed and varies only the representation, so that any
difference is attributable to the representation alone.

The head used for each task is whichever head that task's model selected, read
from the model's artifact, so the comparison is run on the classifier the study
actually deploys.

Run:  .venv/bin/python -m analysis.representation
Output: results/tables/t411_representation.csv    (Table 4.4 in the thesis)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from shared.paths import CV, ensure_dirs
from shared.data import load_splits, raw_text
from shared.encoders import build_embeddings, encode_lsa
from shared.heads import get_head
from shared.metrics import save_table
from shared.artifacts import load_artifact

TASKS = (("Ticket classification", "Category", "classification"),
         ("Resolver routing", "Assigned Team", "routing"))


def main():
    ensure_dirs()
    tr, te, meta = load_splits()
    print(f"{'='*72}\nObjective 2 / RQ2 -- contextual vs lexical representation\n{'='*72}")

    E_tr, _, enc_name, dim, _ = build_embeddings(tr, te)
    L_tr, _, lsa_name, lsa_dim = encode_lsa(raw_text(tr), raw_text(te))
    print(f"  contextual: {enc_name} ({dim}d)")
    print(f"  lexical   : {lsa_name}\n")

    rows = []
    for task, ycol, model_name in TASKS:
        y = tr[ycol].values
        try:
            _, m = load_artifact(model_name)
            hname = m.get("head")
        except SystemExit:
            print(f"  [skip] {task}: run the {model_name} model first")
            continue
        name, est, grid = get_head(hname, len(np.unique(y)))
        n_jobs = 1 if name == "XGBoost" else -1
        for label, X in (("Contextual sentence embedding", E_tr),
                         ("Lexical TF-IDF → LSA", L_tr)):
            sc = float(np.mean(cross_val_score(est, X, y, cv=CV,
                                               scoring="f1_macro", n_jobs=n_jobs)))
            rows.append({"Task": task, "Representation": label, "Head": hname,
                         "Dimensions": X.shape[1], "CV F1 (macro)": round(sc, 4)})
            print(f"  {task:24s} {label:32s} cvF1={sc:.4f}", flush=True)

    if rows:
        save_table(pd.DataFrame(rows), "t411_representation.csv",
                   "Table 4.4 -- contextual vs lexical representation (Objective 2)")


if __name__ == "__main__":
    main()
