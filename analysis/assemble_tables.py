"""
Assemble the per-model outputs into the Chapter Four tables.

Each independent model writes only its own rows, to `_m1_*.csv`, `_m2_*.csv`,
`_m3_*.csv`. That is what keeps them independently runnable: no model has to
know what the others produced. This step stitches those fragments into the
numbered tables the thesis actually cites, and into the paired-prediction file
that analysis/compare.py needs for the McNemar tests.

Run it after any model is re-run, and before compare.py / write_chapters.py.
"""
import os, json
import numpy as np
import pandas as pd

from shared.paths import TABLES, PREDICTIONS, ARTIFACTS, ensure_dirs
from shared.artifacts import load_artifact


def _read(name):
    p = os.path.join(TABLES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def _concat(names, out, caption):
    parts = [d for d in (_read(n) for n in names) if d is not None]
    if not parts:
        print(f"  [skip] {out}: no fragments present")
        return None
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(TABLES, out), index=False)
    print(f"  {out:46s} <- {len(parts)} fragment(s), {len(df)} row(s)")
    return df


def main():
    ensure_dirs()
    print("assembling Chapter Four tables from the per-model fragments:")

    _concat(["_m1_classification.csv", "_m2_routing.csv"],
            "t44_armB_classification_routing.csv", "Table 4.7 in the thesis")
    _concat(["_m1_ablation.csv", "_m2_ablation.csv"],
            "t410_feature_ablation.csv", "Table 4.3 in the thesis")
    _concat(["_m1_heads.csv", "_m2_heads.csv"],
            "t412_armB_head_selection.csv", "Table 4.6 in the thesis")
    _concat(["_m3_recommendation.csv"],
            "t46_armB_recommendation.csv", "Table 4.15 in the thesis")

    # ---- paired predictions + merged metadata for compare.py --------------
    preds, meta = {}, {}
    for model, key in (("classification", "Ticket classification"),
                       ("routing", "Resolver routing")):
        try:
            art, m = load_artifact(model)
        except SystemExit:
            print(f"  [skip] {model} artifact absent")
            continue
        preds[f"B|{key}"] = art["test_pred"].astype(str)
        meta.update({k: v for k, v in m.items() if k not in meta})
    try:
        art, m = load_artifact("recommendation")
        preds["B|rec|hit1"] = art["hit1"].astype(int)
        meta.update({k: v for k, v in m.items() if k not in meta})
    except SystemExit:
        print("  [skip] recommendation artifact absent")

    if preds:
        np.savez(os.path.join(PREDICTIONS, "arm_b.npz"),
                 **{k: np.asarray(v, dtype=object) for k, v in preds.items()})
        json.dump(meta, open(os.path.join(PREDICTIONS, "arm_b_meta.json"), "w"), indent=2)
        print(f"  arm_b.npz / arm_b_meta.json                    <- {len(preds)} prediction set(s)")

    print("\nnext: .venv/bin/python -m analysis.compare")


if __name__ == "__main__":
    main()
