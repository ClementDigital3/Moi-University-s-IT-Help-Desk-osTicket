"""
The abstention trade-off, and what declining buys.

Section 5.4.2 recommends investigating abstention: "a model that declines to
predict and instead requests clarification may be more useful operationally than
one that always answers." This quantifies that for resolver routing, which is
the decision where it matters most -- Table 4.12 shows the first-line team at an
F1 of 0.09, so the model is not merely uncertain about those tickets, it is
reliably wrong about them.

For each confidence threshold it reports coverage, accuracy on the answered
tickets, accuracy on the declined ones, and how far the declined pile is
enriched for the first-line team. If declining concentrates first-line work,
then a decline is not a failure to answer: it is the model identifying the
tickets that belong at first line.

Run:    .venv/bin/python -m analysis.abstention
Output: results/tables/t417_abstention.csv
"""
import os
import numpy as np
import pandas as pd
import joblib

from shared.paths import TABLES, ROOT, ensure_dirs
from shared.data import load_splits
from shared.encoders import build_embeddings
from shared.context import build_blocks, assemble
from shared.artifacts import load_artifact
from shared.serving import class_scores
from shared.metrics import save_table

FIRST_LINE = "Help Desk (Tier 1)"
GRID = [0.00, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def main():
    ensure_dirs()
    tr, te, _ = load_splits()
    b = joblib.load(os.path.join(ROOT, "results", "deploy", "routing.joblib"))
    up, _ = load_artifact("classification")

    E_tr, E_te, *_ = build_embeddings(tr, te)
    blocks, _ = build_blocks(tr, te, E_tr, E_te)
    blocks["cat"] = (b["fitter"].apply_extra("cat", up["oof_proba"].astype(float)),
                     b["fitter"].apply_extra("cat", up["test_proba"].astype(float)))
    _, X = assemble(blocks, b["keys"], b["alpha"])

    classes = list(b["classes"])
    y = te["Assigned Team"].values
    conf = np.empty(X.shape[0])
    pred = np.empty(X.shape[0], dtype=object)
    for i in range(X.shape[0]):
        s, _ = class_scores(b["model"], X[i:i + 1], classes)
        conf[i], pred[i] = s.max(), classes[int(s.argmax())]
    ok = pred == y
    base_rate = float((y == FIRST_LINE).mean())

    rows = []
    for t in GRID:
        ans = conf >= t
        rows.append({
            "Threshold": t,
            "Answered %": round(100 * ans.mean(), 1),
            "Accuracy (answered)": round(float(ok[ans].mean()), 4) if ans.any() else None,
            "Accuracy (declined)": round(float(ok[~ans].mean()), 4) if (~ans).any() else None,
            "First-line share of declined %":
                round(100 * float((y[~ans] == FIRST_LINE).mean()), 1) if (~ans).any() else None,
            "Enrichment vs base":
                round(float((y[~ans] == FIRST_LINE).mean()) / base_rate, 2) if (~ans).any() else None,
        })
    df = pd.DataFrame(rows)
    save_table(df, "t417_abstention.csv",
               f"Table 4.17 -- abstention trade-off for resolver routing "
               f"(first-line base rate {100*base_rate:.1f}%)")
    print(f"\n  routing accuracy without abstention: {ok.mean():.4f}")


if __name__ == "__main__":
    main()
