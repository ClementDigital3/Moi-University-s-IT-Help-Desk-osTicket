"""
The hand-off contract between independent models.

Each model writes exactly one artifact to results/artifacts/ and reads only the
artifacts of the models upstream of it. That is what makes the models
independently developable: you can rewrite the routing model from scratch and
the classification model neither knows nor cares, so long as the artifact it
publishes keeps the same shape.

    classification  ->  results/artifacts/classification.npz
                        test_pred    predicted category per held-out ticket
                        test_proba   class distribution per held-out ticket
                        oof_proba    OUT-OF-FOLD class distribution per TRAINING
                                     ticket -- see the leakage note below
                        classes      the class order both matrices use

    routing         ->  results/artifacts/routing.npz
    recommendation  ->  results/artifacts/recommendation.npz

Leakage note. Downstream models are cascaded on the category, so they must never
see the TRUE category of a training ticket. `oof_proba` is produced by
cross_val_predict, meaning each training row's category distribution comes from
folds that excluded that row. A downstream model fitted on `oof_proba` therefore
meets category context of exactly the quality it will meet at inference, and its
reported performance is honest.
"""
import os, json
import numpy as np

from .paths import ARTIFACTS


def _paths(name):
    os.makedirs(ARTIFACTS, exist_ok=True)
    return (os.path.join(ARTIFACTS, f"{name}.npz"),
            os.path.join(ARTIFACTS, f"{name}.json"))


def save_artifact(name, meta=None, **arrays):
    """Publish this model's outputs for downstream models and for compare.py."""
    npz, js = _paths(name)
    np.savez_compressed(
        npz, **{k: np.asarray(v, dtype=object) if _is_str(v) else np.asarray(v)
                for k, v in arrays.items()})
    json.dump(meta or {}, open(js, "w"), indent=2)
    print(f"  artifact -> {os.path.relpath(npz, os.path.dirname(ARTIFACTS))}"
          f"  [{', '.join(arrays)}]")
    return npz


def load_artifact(name):
    """
    Read an upstream model's outputs.

    Fails loudly with the command to run, rather than silently degrading, so a
    missing dependency is never mistaken for a modelling result.
    """
    npz, js = _paths(name)
    if not os.path.exists(npz):
        raise SystemExit(
            f"missing upstream artifact: {npz}\n"
            f"run the '{name}' model first:\n"
            f"    .venv/bin/python -m models.{name}.run")
    z = np.load(npz, allow_pickle=True)
    meta = json.load(open(js)) if os.path.exists(js) else {}
    return {k: z[k] for k in z.files}, meta


def _is_str(v):
    a = np.asarray(v)
    return a.dtype.kind in ("U", "S", "O")
