"""
Head-to-head comparison -- the study's central experiment.

    ARM A   one end-to-end model.
            A single TF-IDF representation is fitted once and a combined
            (voted) classifier stack reads it for classification and routing,
            with lexical cosine retrieval for recommendation. One pipeline,
            every decision.

    ARM B   three dedicated context-aware models.
            B1, B2 and B3 are built, represented and tuned independently, each
            conditioned on the contextual and metadata signals relevant to its
            own decision, with B2 cascaded on B1.

For each of the three help desk decisions this script asks whether the
task-specific, context-aware treatment beats the single end-to-end pipeline --
and whether any observed difference is distinguishable from sampling noise.

Both arms were evaluated on the identical held-out partition written by
preprocess.py, so every comparison is paired and McNemar's exact test applies.
"""
import os, json
import numpy as np
import pandas as pd

from shared.paths import TABLES, PREDICTIONS as PRED, ROOT
from shared.data import load_splits
from shared.metrics import save_table, mcnemar, bootstrap_ci

TASKS = (("Ticket classification", "Category"),
         ("Resolver routing", "Assigned Team"))


def load_arm(name):
    z = np.load(os.path.join(PRED, f"{name}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    tr, te, meta = load_splits()
    A, B = load_arm("arm_a"), load_arm("arm_b")
    bmeta = json.load(open(os.path.join(PRED, "arm_b_meta.json")))

    armA = pd.read_csv(os.path.join(TABLES, "t43_armA_classification_routing.csv"))
    armB = pd.read_csv(os.path.join(TABLES, "t44_armB_classification_routing.csv"))
    recA = pd.read_csv(os.path.join(TABLES, "t45_armA_recommendation.csv")).iloc[0]
    recB = pd.read_csv(os.path.join(TABLES, "t46_armB_recommendation.csv")).iloc[0]

    head, stats, branch = [], [], []

    # ---------------- classification & routing ----------------------------
    for task, ycol in TASKS:
        y = te[ycol].values
        sub = armA[armA["Task"] == task]

        # Arm A's DEPLOYED decision is the combined vote: that is the end-to-end
        # model. Its best single branch is reported separately, for context.
        comb = sub[sub["Arm"] == "A (combined)"]
        a_row = comb.loc[comb["F1 (macro)"].idxmax()]
        br = sub[sub["Arm"] == "A (branch)"]
        b_best = br.loc[br["F1 (macro)"].idxmax()]
        b_row = armB[armB["Task"] == task].iloc[0]

        pa = A[f"A|{task}|{a_row['Model']}"].astype(str)
        pb = B[f"B|{task}"].astype(str)
        loA, loB = bootstrap_ci(y, pa), bootstrap_ci(y, pb)
        bb, cc, p = mcnemar(y, pa, pb)

        maj = pd.Series(tr[ycol]).mode()[0]
        head.append({
            "Task": task,
            "Arm A model": a_row["Model"],
            "A accuracy": a_row["Accuracy"], "A macro F1": a_row["F1 (macro)"],
            "A 95% CI": f"[{loA[0]:.3f}, {loA[1]:.3f}]",
            "Arm B model": b_row["Model"],
            "B accuracy": b_row["Accuracy"], "B macro F1": b_row["F1 (macro)"],
            "B 95% CI": f"[{loB[0]:.3f}, {loB[1]:.3f}]",
            "Δ accuracy (B−A)": round(b_row["Accuracy"] - a_row["Accuracy"], 4),
            "Δ macro F1 (B−A)": round(b_row["F1 (macro)"] - a_row["F1 (macro)"], 4),
            "Majority baseline": round(float((y == maj).mean()), 4),
        })
        branch.append({
            "Task": task,
            "Arm A best single branch": b_best["Model"],
            "Branch macro F1": b_best["F1 (macro)"],
            "Arm A combined macro F1": a_row["F1 (macro)"],
            "Combining gain": round(a_row["F1 (macro)"] - b_best["F1 (macro)"], 4),
            "Arm B macro F1": b_row["F1 (macro)"],
            "Arm B head": b_row.get("Head", ""),
            "Arm B feature set": b_row.get("Feature set", ""),
        })
        stats.append({
            "Task": task, "Test": "McNemar (exact)",
            "A right / B wrong (b)": bb, "A wrong / B right (c)": cc,
            "p-value": round(p, 5),
            "Significant at .05": "Yes" if p < 0.05 else "No",
            "Favours": ("Arm B (task-specific)" if cc > bb else
                        "Arm A (end-to-end)" if bb > cc else "tie")
                       if p < 0.05 else "—",
        })

    # ---------------- recommendation --------------------------------------
    head.append({
        "Task": "Resolution recommendation",
        "Arm A model": recA["Model"], "A accuracy": recA["Top-1"],
        "A macro F1": recA["MRR"], "A 95% CI": "—",
        "Arm B model": recB["Model"], "B accuracy": recB["Top-1"],
        "B macro F1": recB["MRR"], "B 95% CI": "—",
        "Δ accuracy (B−A)": round(recB["Top-1"] - recA["Top-1"], 4),
        "Δ macro F1 (B−A)": round(recB["MRR"] - recA["MRR"], 4),
        "Majority baseline": "—",
    })
    ha, hb = A["A|rec|hit1"].astype(int), B["B|rec|hit1"].astype(int)
    bb, cc, p = mcnemar(np.ones_like(ha), ha, hb)
    stats.append({"Task": "Resolution recommendation (Top-1 hit)",
                  "Test": "McNemar (exact)", "A right / B wrong (b)": bb,
                  "A wrong / B right (c)": cc, "p-value": round(p, 5),
                  "Significant at .05": "Yes" if p < 0.05 else "No",
                  "Favours": ("Arm B (task-specific)" if cc > bb
                              else "Arm A (end-to-end)") if p < 0.05 else "—"})

    hd = pd.DataFrame(head)
    save_table(hd, "t47_head_to_head.csv",
               "Table 4.7 -- Arm A (end-to-end) vs Arm B (task-specific), held-out test set")
    save_table(pd.DataFrame(stats), "t48_significance.csv",
               "Table 4.8 -- Paired significance tests (McNemar, exact)")
    save_table(pd.DataFrame(branch), "t414_branch_vs_combined.csv",
               "Table 4.14 -- Arm A combining gain, and the Arm B configuration it faces")

    # ---------------- operational comparison ------------------------------
    a_fit = float(armA.groupby("Task")["Fit time (s)"].max().sum())
    ops = pd.DataFrame([
        {"Dimension": "Models fitted for deployment",
         "Arm A (end-to-end)": "5 base classifiers × 2 tasks, combined by vote",
         "Arm B (task-specific)": "1 encoder + 1 head per label task + 1 index"},
        {"Dimension": "Representations maintained",
         "Arm A (end-to-end)": "1 shared TF-IDF vocabulary (+1 for retrieval)",
         "Arm B (task-specific)": "1 shared sentence-embedding space"},
        {"Dimension": "Inference per ticket",
         "Arm A (end-to-end)": "1 vectorise → 10 model calls → 2 votes → 1 search",
         "Arm B (task-specific)": "1 encode → B1 → B2 (consumes B1) → 1 search"},
        {"Dimension": "Training cost (s)",
         "Arm A (end-to-end)": f"{a_fit:.1f} (supervised fits)",
         "Arm B (task-specific)": f"{bmeta['encode_seconds']:.1f} encode + head fits"},
        {"Dimension": "Adding a new category or resolver",
         "Arm A (end-to-end)": "refit all 5 base classifiers for that task",
         "Arm B (task-specific)": "refit one head; the encoder is unchanged"},
        {"Dimension": "Cold-start on unseen vocabulary",
         "Arm A (end-to-end)": "out-of-vocabulary terms are simply dropped",
         "Arm B (task-specific)": "encoder generalises from pretraining"},
        {"Dimension": "Explanation offered to the resolver",
         "Arm A (end-to-end)": "vote margin across base classifiers",
         "Arm B (task-specific)": "the neighbouring tickets and their resolutions"},
        {"Dimension": "Failure coupling",
         "Arm A (end-to-end)": "tasks independent; a bad category does not affect routing",
         "Arm B (task-specific)":
             f"cascaded: routing inherits B1 error (OOF agreement "
             f"{bmeta['oof_category_agreement']:.3f})"},
    ])
    save_table(ops, "t49_operational.csv",
               "Table 4.9 -- Operational comparison (the Sec. 2.11 evaluation gap)")

    # ---------------- verdict ---------------------------------------------
    print("\n" + "=" * 72 + "\nVERDICT\n" + "=" * 72)
    for _, r in hd.iterrows():
        d = r["Δ macro F1 (B−A)"]
        sig = [s for s in stats if s["Task"].startswith(r["Task"])]
        pv = sig[0]["p-value"] if sig else float("nan")
        which = ("Arm B (task-specific)" if d > 0 else
                 "Arm A (end-to-end)" if d < 0 else "tie")
        mark = "significant" if (isinstance(pv, float) and pv < 0.05) else "not significant"
        print(f"  {r['Task']:28s} Δ={d:+.4f}  {which:24s} ({mark}, p={pv})")


if __name__ == "__main__":
    main()
