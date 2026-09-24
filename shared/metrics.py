"""
Shared evaluation layer -- implements Table 3.3 of the thesis.

Classification / resolver routing : accuracy, precision, recall, F1,
                                    macro and weighted F1, confusion matrix,
                                    correct-routing rate, cross-validation.
Resolution recommendation         : Top-1, Top-3, Top-5, Precision@K, Recall@K,
                                    Mean Reciprocal Rank.

Also provides the paired significance tests used to decide whether an observed
difference between two models is real or sampling noise.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

from .paths import TABLES, SEED


# --------------------------------------------------------------------------
# Classification / routing
# --------------------------------------------------------------------------
def clf_metrics(y_true, y_pred, task, model):
    """One row of Table 4.3 / 4.4."""
    acc = accuracy_score(y_true, y_pred)
    pm, rm, f1m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    pw, rw, f1w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    row = {"Task": task, "Model": model,
           "Accuracy": round(acc, 4),
           "Precision (macro)": round(pm, 4), "Recall (macro)": round(rm, 4),
           "F1 (macro)": round(f1m, 4), "F1 (weighted)": round(f1w, 4)}
    if task == "Resolver routing":
        # Sec. 2.6 / Table 3.3: routing is judged by correct-routing rate, i.e.
        # the proportion of tickets reaching the correct resolver first time.
        row["Correct-routing rate"] = round(acc, 4)
    return row


def per_class_report(y_true, y_pred, labels=None):
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                zero_division=0)
    rows = []
    for k, v in rep.items():
        if isinstance(v, dict) and k not in ("macro avg", "weighted avg", "accuracy"):
            rows.append({"Class": k, "Precision": round(v["precision"], 3),
                         "Recall": round(v["recall"], 3), "F1": round(v["f1-score"], 3),
                         "Support": int(v["support"])})
    return pd.DataFrame(rows).sort_values("Support", ascending=False)


def conf_matrix_df(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


# --------------------------------------------------------------------------
# Recommendation  (Objective 4 / RQ4)
# --------------------------------------------------------------------------
def rank_metrics(ranked_pids, true_pids, ks=(1, 3, 5, 10)):
    """
    ranked_pids : list per query of latent problem ids of the retrieved
                  historical tickets, best first.
    true_pids   : the query's own latent problem id.

    A retrieved ticket is relevant iff it shares the query's latent problem id.
    Precision@K is computed over the K retrieved; Recall@K is capped at the
    number of relevant items available, so it is not penalised for K < n_rel.
    """
    out = {f"Top-{k}": 0.0 for k in ks}
    out.update({f"Precision@{k}": 0.0 for k in ks})
    out.update({f"Recall@{k}": 0.0 for k in ks})
    rr = []
    n = len(true_pids)
    for ranked, truth in zip(ranked_pids, true_pids):
        hits = [1 if p == truth else 0 for p in ranked]
        for k in ks:
            topk = hits[:k]
            out[f"Top-{k}"] += 1.0 if any(topk) else 0.0
            out[f"Precision@{k}"] += sum(topk) / k
            out[f"Recall@{k}"] += sum(topk) / max(1, min(k, len(hits)))
        try:
            rr.append(1.0 / (hits.index(1) + 1))
        except ValueError:
            rr.append(0.0)
    for k in out:
        out[k] = round(out[k] / n, 4)
    out["MRR"] = round(float(np.mean(rr)), 4)
    return out


# --------------------------------------------------------------------------
# Paired significance testing
# --------------------------------------------------------------------------
def mcnemar(y_true, pred_a, pred_b):
    """
    Exact McNemar test on the paired correct/incorrect outcomes of two models
    evaluated on the identical held-out set. Returns (b, c, p).
      b = A right, B wrong    c = A wrong, B right
    """
    from scipy.stats import binomtest
    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 1.0
    p = binomtest(b, b + c, 0.5).pvalue
    return b, c, float(p)


def bootstrap_ci(y_true, y_pred, stat=accuracy_score, n_boot=2000, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = [stat(y_true[i], y_pred[i]) for i in
            (rng.integers(0, n, n) for _ in range(n_boot))]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


def save_table(df, name, caption=""):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name)
    df.to_csv(path, index=False)
    if caption:
        print(f"\n--- {caption} ---")
    print(df.to_string(index=False))
    return path
