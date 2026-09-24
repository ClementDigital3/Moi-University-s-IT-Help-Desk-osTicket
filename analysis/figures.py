"""
Figures for Chapter Four. Written to results/figures/ at 300 dpi for insertion
into the thesis document.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.paths import TABLES, FIGURES, ROOT
from shared.data import load_splits

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.autolayout": True,
})
C_A, C_B, C_COMB = "#4C72B0", "#DD8452", "#2F4B7C"


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    p = os.path.join(FIGURES, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def fig_class_dist():
    df = pd.read_csv(os.path.join(TABLES, "t42_classdist.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, lab in zip(axes, ["Category", "Resolver team"]):
        d = df[(df["Label type"] == lab) & (~df["Class"].str.startswith("--"))]
        d = d.sort_values("n")
        ax.barh(d["Class"], d["n"], color=C_A)
        ax.set_title(f"{lab} distribution", fontsize=10)
        ax.set_xlabel("tickets")
        for y, (n, pc) in enumerate(zip(d["n"], d["% of corpus"])):
            ax.text(n + max(d["n"]) * .01, y, f"{n} ({pc}%)", va="center", fontsize=7.5)
        ax.set_xlim(0, max(d["n"]) * 1.25)
    fig.suptitle("Class distribution after screening", y=1.04, fontsize=11)
    save(fig, "fig41_class_distribution.png")


def fig_armA_models():
    df = pd.read_csv(os.path.join(TABLES, "t43_armA_classification_routing.csv"))
    tasks = df["Task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(11.5, 4.0), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), tasks):
        d = df[df["Task"] == t].sort_values("F1 (macro)")
        cols = [C_COMB if str(a).startswith("A (combined)") else C_A for a in d["Arm"]]
        ax.barh(d["Model"], d["F1 (macro)"], color=cols)
        for y, v in enumerate(d["F1 (macro)"]):
            ax.text(v + .006, y, f"{v:.3f}", va="center", fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("macro F1")
        ax.set_xlim(0, min(1.0, d["F1 (macro)"].max() * 1.18))
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_A),
               plt.Rectangle((0, 0), 1, 1, color=C_COMB)]
    np.atleast_1d(axes)[0].legend(handles, ["individual branch", "combined (voted)"],
                                  frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Arm A — the end-to-end model and its individual branches",
                 y=1.04, fontsize=11)
    save(fig, "fig42_armA_model_comparison.png")


def fig_head_to_head():
    hd = pd.read_csv(os.path.join(TABLES, "t47_head_to_head.csv"))
    clf = hd[hd["Task"] != "Resolution recommendation"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(clf)); w = 0.36
    ax.bar(x - w/2, clf["A macro F1"], w, label="Arm A (end-to-end)", color=C_A)
    ax.bar(x + w/2, clf["B macro F1"], w, label="Arm B (task-specific)", color=C_B)
    for i, (a, b) in enumerate(zip(clf["A macro F1"], clf["B macro F1"])):
        ax.text(i - w/2, a + .008, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, b + .008, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(clf["Task"])
    ax.set_ylabel("macro F1")
    ax.set_ylim(0, max(clf[["A macro F1", "B macro F1"]].max()) * 1.2)
    ax.legend(frameon=False)
    ax.set_title("End-to-end vs task-specific models, held-out test set", fontsize=11)
    save(fig, "fig43_head_to_head.png")


def fig_topk():
    a = pd.read_csv(os.path.join(TABLES, "t45_armA_recommendation.csv")).iloc[0]
    b = pd.read_csv(os.path.join(TABLES, "t46_armB_recommendation.csv")).iloc[0]
    ks = [1, 3, 5, 10]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(ks, [a[f"Top-{k}"] for k in ks], "o-", color=C_A, label="Arm A (lexical)")
    axes[0].plot(ks, [b[f"Top-{k}"] for k in ks], "s-", color=C_B, label="Arm B (context-aware)")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Top-K accuracy")
    axes[0].set_title("Top-K accuracy", fontsize=10); axes[0].set_xticks(ks)
    axes[0].legend(frameon=False); axes[0].set_ylim(0, 1.02)
    names = ["Top-1", "Top-5", "MRR"]
    x = np.arange(len(names)); w = .36
    axes[1].bar(x - w/2, [a[n] for n in names], w, color=C_A, label="Arm A")
    axes[1].bar(x + w/2, [b[n] for n in names], w, color=C_B, label="Arm B")
    for i, n in enumerate(names):
        axes[1].text(i - w/2, a[n] + .012, f"{a[n]:.3f}", ha="center", fontsize=8)
        axes[1].text(i + w/2, b[n] + .012, f"{b[n]:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylim(0, 1.15); axes[1].legend(frameon=False)
    axes[1].set_title("Ranking quality", fontsize=10)
    fig.suptitle("Historical-resolution recommendation (Objective 4)",
                 y=1.05, fontsize=11)
    save(fig, "fig44_recommendation.png")


def fig_confusion():
    files = [("t4_armA_route_confusion.csv", "Arm A (end-to-end)"),
             ("t4_armB_route_confusion.csv", "Arm B (task-specific)")]
    have = [(f, t) for f, t in files if os.path.exists(os.path.join(TABLES, f))]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(6.2 * len(have), 5.2))
    for ax, (f, t) in zip(np.atleast_1d(axes), have):
        cm = pd.read_csv(os.path.join(TABLES, f), index_col=0)
        norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(cm.columns)))
        ax.set_xticklabels([c.replace(" ", "\n") for c in cm.columns], fontsize=6.5)
        ax.set_yticks(range(len(cm.index)))
        ax.set_yticklabels(cm.index, fontsize=7)
        ax.set_title(f"{t} — resolver routing", fontsize=10)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.grid(False)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = norm.iloc[i, j]
                if v > .01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if v > .55 else "black")
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=.75, label="row-normalised")
    fig.suptitle("Routing confusion matrices", y=1.0, fontsize=11)
    save(fig, "fig45_routing_confusion.png")


def fig_ablation():
    """Objective 1 / RQ1 -- what each context block is actually worth."""
    df = pd.read_csv(os.path.join(TABLES, "t410_feature_ablation.csv"))
    tasks = df["Task"].unique()
    fig, axes = plt.subplots(1, len(tasks), figsize=(11.5, 3.9))
    for ax, t in zip(np.atleast_1d(axes), tasks):
        d = df[df["Task"] == t]
        ax.plot(range(len(d)), d["CV F1 (macro)"], "o-", color=C_A, lw=1.8)
        for i, v in enumerate(d["CV F1 (macro)"]):
            ax.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([s.replace("+ ", "+\n").replace(" only", "\nonly")
                            for s in d["Feature set"]], fontsize=7.5)
        ax.set_title(t, fontsize=10)
        ax.set_ylabel("cross-validated macro F1")
        lo, hi = d["CV F1 (macro)"].min(), d["CV F1 (macro)"].max()
        pad = max((hi - lo) * 0.6, 0.012)
        ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle("Cumulative contribution of each context block (Objective 1)",
                 y=1.04, fontsize=11)
    save(fig, "fig46_feature_ablation.png")


def fig_representation():
    """Objective 2 / RQ2 -- contextual embedding vs lexical representation."""
    df = pd.read_csv(os.path.join(TABLES, "t411_representation.csv"))
    tasks = list(df["Task"].unique())
    reps = list(df["Representation"].unique())
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    x = np.arange(len(tasks)); w = 0.36
    for i, (rep, col) in enumerate(zip(reps, [C_B, C_A])):
        vals = [float(df[(df["Task"] == t) & (df["Representation"] == rep)]
                      ["CV F1 (macro)"].iloc[0]) for t in tasks]
        off = (i - (len(reps) - 1) / 2) * w
        ax.bar(x + off, vals, w, label=rep, color=col)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + .008, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylabel("cross-validated macro F1")
    ax.set_ylim(0, min(1.0, df["CV F1 (macro)"].max() * 1.25))
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Contextual vs lexical representation, same classifier head",
                 fontsize=11)
    save(fig, "fig47_representation.png")


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    print("writing figures:")
    for f in (fig_class_dist, fig_armA_models, fig_head_to_head, fig_topk,
              fig_confusion, fig_ablation, fig_representation):
        try:
            f()
        except Exception as e:
            print(f"  [skip] {f.__name__}: {type(e).__name__}: {e}")
