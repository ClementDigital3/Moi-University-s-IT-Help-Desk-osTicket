"""
Collect every Chapter Four result into one JSON payload for the dashboard.

Kept separate from the page builder so the data contract is inspectable on its
own: run this module directly to dump the payload and see exactly what the
dashboard is drawing from.
"""
import os, json
import numpy as np
import pandas as pd

from shared.paths import TABLES, PREDICTIONS, ARTIFACTS, DATA_PROC, ROOT


def _csv(name):
    p = os.path.join(TABLES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _recs(df, cols=None):
    if df is None:
        return []
    d = df[[c for c in cols if c in df.columns]] if cols else df
    return json.loads(d.to_json(orient="records"))


def _json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else (default or {})


def collect():
    meta = _json(os.path.join(DATA_PROC, "meta.json"))
    bmeta = _json(os.path.join(PREDICTIONS, "arm_b_meta.json"))

    head = _csv("t47_head_to_head.csv")
    sig = _csv("t48_significance.csv")
    dist = _csv("t42_classdist.csv")

    # headline: one entry per decision, each with its OWN metric named, so the
    # dashboard never implies macro F1 and MRR are the same scale
    decisions = []
    if head is not None and sig is not None:
        for _, r in head.iterrows():
            task = r["Task"]
            s = sig[sig["Task"].str.startswith(task)]
            s = s.iloc[0] if len(s) else None
            is_rec = task == "Resolution recommendation"
            d = float(r["Δ macro F1 (B−A)"])
            decisions.append({
                "task": task,
                "metric": "Mean Reciprocal Rank" if is_rec else "Macro F1",
                "a_model": r["Arm A model"], "b_model": r["Arm B model"],
                "a": float(r["A macro F1"]), "b": float(r["B macro F1"]),
                "a_secondary": float(r["A accuracy"]), "b_secondary": float(r["B accuracy"]),
                "secondary_metric": "Top-1" if is_rec else "Accuracy",
                "delta": round(d, 4),
                "winner": ("Arm B" if d > 0 else "Arm A" if d < 0 else "tie"),
                "p": float(s["p-value"]) if s is not None else None,
                "significant": (str(s["Significant at .05"]).strip().lower() == "yes")
                               if s is not None else None,
                "b_disagree": int(s["A right / B wrong (b)"]) if s is not None else None,
                "c_disagree": int(s["A wrong / B right (c)"]) if s is not None else None,
                "baseline": (None if is_rec else float(r["Majority baseline"])),
            })

    def dist_for(label):
        if dist is None:
            return []
        d = dist[(dist["Label type"] == label) & (~dist["Class"].str.startswith("--"))]
        return [{"label": x["Class"], "n": int(x["n"]), "pct": float(x["% of corpus"])}
                for _, x in d.iterrows()]

    def imbalance(label):
        if dist is None:
            return None
        v = dist[(dist["Label type"] == label) & (dist["Class"].str.startswith("--"))]["n"]
        return float(v.iloc[0]) if len(v) else None

    # confusion matrices, row-normalised
    def confusion(fname):
        p = os.path.join(TABLES, fname)
        if not os.path.exists(p):
            return None
        cm = pd.read_csv(p, index_col=0)
        norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        return {"labels": [str(c) for c in cm.columns],
                "rows": [str(i) for i in cm.index],
                "counts": cm.values.astype(int).tolist(),
                "norm": np.round(norm.values, 4).tolist()}

    armA = _csv("t43_armA_classification_routing.csv")
    recA = _csv("t45_armA_recommendation.csv")
    recB = _csv("t46_armB_recommendation.csv")

    payload = {
        "corpus": {
            "n_clean": meta.get("n_clean"), "n_train": meta.get("n_train"),
            "n_test": meta.get("n_test"),
            "categories": meta.get("categories", []), "teams": meta.get("teams", []),
            "mean_tokens": meta.get("mean_tokens"), "median_tokens": meta.get("median_tokens"),
            "screening": _recs(_csv("t41_screening.csv")),
            "category_dist": dist_for("Category"),
            "team_dist": dist_for("Resolver team"),
            "category_imbalance": imbalance("Category"),
            "team_imbalance": imbalance("Resolver team"),
        },
        "encoder": {
            "name": bmeta.get("encoder"), "dim": bmeta.get("dim"),
            "encode_seconds": bmeta.get("encode_seconds"),
            "oof_category_agreement": bmeta.get("oof_category_agreement"),
            "blend_w_semantic": bmeta.get("blend_w_semantic"),
        },
        "decisions": decisions,
        "significance": _recs(sig),
        "ablation": _recs(_csv("t410_feature_ablation.csv"),
                          ["Task", "Feature set", "Blocks", "Dimensions",
                           "Block weight", "CV F1 (macro)", "Δ vs previous"]),
        "representation": _recs(_csv("t411_representation.csv")),
        "arm_a_models": _recs(armA, ["Task", "Model", "Accuracy", "F1 (macro)",
                                     "F1 (weighted)", "CV F1 (macro)", "Fit time (s)", "Arm"]),
        "arm_b_models": _recs(_csv("t44_armB_classification_routing.csv"),
                              ["Task", "Head", "Feature set", "Dimensions",
                               "Accuracy", "F1 (macro)", "CV F1 (macro)", "Fit time (s)"]),
        "arm_b_heads": _recs(_csv("t412_armB_head_selection.csv")),
        "recommendation_variants": _recs(_csv("t413_recommendation_variants.csv")),
        "topk": {
            "ks": [1, 3, 5, 10],
            "arm_a": ([float(recA.iloc[0][f"Top-{k}"]) for k in (1, 3, 5, 10)]
                      if recA is not None else []),
            "arm_b": ([float(recB.iloc[0][f"Top-{k}"]) for k in (1, 3, 5, 10)]
                      if recB is not None else []),
            "a_model": recA.iloc[0]["Model"] if recA is not None else "",
            "b_model": recB.iloc[0]["Model"] if recB is not None else "",
            "a_mrr": float(recA.iloc[0]["MRR"]) if recA is not None else None,
            "b_mrr": float(recB.iloc[0]["MRR"]) if recB is not None else None,
        },
        "operational": _recs(_csv("t49_operational.csv")),
        "combining": _recs(_csv("t414_branch_vs_combined.csv")),
        "confusion": {
            "arm_a_routing": confusion("t4_armA_route_confusion.csv"),
            "arm_b_routing": confusion("t4_armB_route_confusion.csv"),
        },
        "per_class": {
            "arm_a_routing": _recs(_csv("t4_armA_route_perclass.csv")),
            "arm_b_routing": _recs(_csv("t4_armB_route_perclass.csv")),
        },
    }
    return payload


if __name__ == "__main__":
    import sys
    p = collect()
    json.dump(p, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout", "w"), indent=2)
