"""
Ranking distinct resolutions, rather than the tickets that carry them.

Objective Four retrieves similar resolved tickets. But a resolver does not want
tickets, they want the fix -- and of the five tickets retrieved, only about two
carry different fixes; for roughly a quarter of queries all five are the same
fix repeated. Collapsing to distinct resolutions turns a list of near-copies
into a short list of genuine options.

Collapsing also makes a choice available that ranking tickets does not: when
several retrieved tickets agree on the same fix, that agreement is evidence.
This compares three ways of ordering the collapsed list and reports whether the
agreement is worth anything.

Run:    .venv/bin/python -m analysis.resolution_ranking
Output: results/tables/t419_resolution_ranking.csv
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from shared.paths import ROOT, ensure_dirs
from shared.data import load_splits
from shared.encoders import build_embeddings
from shared.metrics import save_table


def main():
    ensure_dirs()
    tr, te, _ = load_splits()
    b = joblib.load(os.path.join(ROOT, "results", "deploy", "recommendation.joblib"))
    E_tr, E_te, *_ = build_embeddings(tr, te)
    keep = tr["res_clean"].fillna("").str.strip().values != ""
    corpus = tr[keep].reset_index(drop=True)

    S = (b["blend_w"] * (E_te @ b["embeddings"].T)
         + (1 - b["blend_w"]) * cosine_similarity(
             b["lex_vectorizer"].transform(te["text"]), b["lex_matrix"]))
    res = corpus["Resolution Notes"].values
    pid = corpus["latent_problem_id"].values
    qpid = te["latent_problem_id"].values

    strategies = {
        "Top single similarity": lambda q, g: S[q, g[0]],
        "Most tickets agreeing": lambda q, g: (len(g), S[q, g[0]]),
        "Summed similarity":     lambda q, g: sum(S[q, j] for j in g),
    }

    rows = []
    for K in (5, 10, 15):
        topk = np.argsort(-S, axis=1)[:, :K]
        distinct = []
        hits = {k: 0 for k in strategies}
        for q in range(len(te)):
            groups = defaultdict(list)
            for j in topk[q]:
                groups[res[j]].append(j)
            distinct.append(len(groups))
            for name, key in strategies.items():
                best = max(groups.values(), key=lambda g: key(q, g))
                hits[name] += int(pid[best[0]] == qpid[q])
        for name in strategies:
            rows.append({
                "Retrieved (K)": K,
                "Ranking": name,
                "Distinct resolutions shown": round(float(np.mean(distinct)), 2),
                "Top-1 applies": round(hits[name] / len(te), 4),
            })
            print(f"  K={K:<3d} {name:24s} top-1 applies {hits[name]/len(te):.4f}"
                  f"   ({np.mean(distinct):.2f} distinct shown)")

    df = pd.DataFrame(rows)
    save_table(df, "t419_resolution_ranking.csv",
               "Table 4.19 -- ranking distinct resolutions (held-out queries)")

    base = df[(df["Retrieved (K)"] == 5) & (df["Ranking"] == "Top single similarity")]["Top-1 applies"].iloc[0]
    best = df.loc[df["Top-1 applies"].idxmax()]
    print(f"\n  ticket-level baseline      {base:.4f}")
    print(f"  best collapsed ranking     {best['Top-1 applies']:.4f}"
          f"  ({best['Ranking']}, K={int(best['Retrieved (K)'])})")
    print(f"  gain                       {best['Top-1 applies'] - base:+.4f}")


if __name__ == "__main__":
    main()
