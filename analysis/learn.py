"""
What accumulation is worth, and the machinery for accumulating.

The help desk record grows. Section 1.1 calls it "an accumulated body of
institutional knowledge", and every ticket a resolver settles adds to it. Two
questions follow, and this module answers both.

  --curve    Is the system data-limited or ceiling-limited? If accuracy is
             still climbing with corpus size, more confirmed tickets will buy
             more performance and the feedback loop is worth running. If it has
             flattened, the limit is the tickets themselves -- the ambiguity
             Chapter Four identifies -- and more of the same will not help.
             Produces Table 4.18.

  --status   What the artefact has collected so far, by provenance.
  --promote  Move recorded decisions into the training store.

The curve is measured on the untouched held-out partition, so it says what
would have happened on data the models never saw, at each corpus size.

Run:  .venv/bin/python -m analysis.learn --curve
"""
import os, argparse
import numpy as np
import pandas as pd

from shared.paths import TABLES, SEED, ensure_dirs
from shared.data import load_splits
from shared.encoders import build_embeddings
from shared.context import build_blocks, assemble
from shared.heads import get_head
from shared.metrics import save_table, clf_metrics
from shared import feedback

FRACTIONS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.85, 1.00]
REPEATS = 3


def curve():
    ensure_dirs()
    tr, te, _ = load_splits()
    E_tr, E_te, *_ = build_embeddings(tr, te)
    blocks, _ = build_blocks(tr, te, E_tr, E_te)
    X_tr, X_te = assemble(blocks, ["text"])

    tasks = [("Ticket classification", "Category", "k-NN (cosine)"),
             ("Resolver routing", "Assigned Team", "Linear SVM")]
    rows = []
    for task, ycol, head in tasks:
        y_tr, y_te = tr[ycol].values, te[ycol].values
        name, est, grid = get_head(head, len(np.unique(y_tr)))
        print(f"\n{task}  ({head})")
        for frac in FRACTIONS:
            scores = []
            n = int(len(y_tr) * frac)
            for rep in range(REPEATS if frac < 1.0 else 1):
                rng = np.random.default_rng(SEED + rep)
                # stratified subsample, so a small corpus still sees every class
                idx = np.concatenate([
                    rng.choice(np.where(y_tr == c)[0],
                               max(1, int(round((y_tr == c).sum() * frac))),
                               replace=False)
                    for c in np.unique(y_tr)])
                m = est.__class__(**est.get_params())
                m.fit(X_tr[idx], y_tr[idx])
                scores.append(clf_metrics(y_te, m.predict(X_te), task, head)["F1 (macro)"])
            rows.append({"Task": task, "Training tickets": n,
                         "Corpus %": round(100 * frac),
                         "Macro F1": round(float(np.mean(scores)), 4),
                         "SD": round(float(np.std(scores)), 4) if len(scores) > 1 else 0.0})
            print(f"  {n:5d} tickets ({100*frac:3.0f}%)  macro F1 {np.mean(scores):.4f}")

    df = pd.DataFrame(rows)
    save_table(df, "t418_learning_curve.csv",
               "Table 4.18 -- what corpus size buys (held-out macro F1)")

    print("\n  marginal value of the last quarter of the corpus:")
    for task in df["Task"].unique():
        d = df[df["Task"] == task].sort_values("Training tickets")
        a = d[d["Corpus %"] == 70]["Macro F1"].iloc[0]
        b = d[d["Corpus %"] == 100]["Macro F1"].iloc[0]
        n = d[d["Corpus %"] == 100]["Training tickets"].iloc[0] - \
            d[d["Corpus %"] == 70]["Training tickets"].iloc[0]
        print(f"    {task:24s} {b - a:+.4f} macro F1 from {n} more tickets "
              f"({(b-a)/max(n,1)*1000:+.4f} per 1,000)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curve", action="store_true", help="measure what corpus size buys")
    ap.add_argument("--status", action="store_true", help="what has been collected")
    ap.add_argument("--promote", action="store_true", help="decisions -> training store")
    a = ap.parse_args()

    if a.promote:
        fresh = feedback.promote()
        print(f"  promoted {len(fresh)} new example(s) into {feedback.STORE}")
    if a.status or a.promote:
        s = feedback.summary()
        print(f"\n  held:                 {s['held']}")
        print(f"  by provenance:        {s['by_provenance'] or '-'}")
        print(f"  trainable (default):  {s['trainable_default_policy']}"
              f"   [corrected + enriched; confirmations excluded by default]")
        print(f"  rejections, no label: {s['rejections_without_a_label']}")
        if s["mean_enrichment_lift"] is not None:
            print(f"  mean enrichment lift: {s['mean_enrichment_lift']:+.3f}")
    if a.curve:
        curve()
    if not (a.curve or a.status or a.promote):
        ap.print_help()


if __name__ == "__main__":
    main()
