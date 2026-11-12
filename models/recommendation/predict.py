"""
Inference for Arm B model B3 — rank resolved tickets by similarity.

`category` narrows the candidates to what B1 predicted, which is the
context-aware variant the evaluation selected. Pass None to retrieve over the
whole archive.
"""
import os, re
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

from shared.serving import ticket_frame, encode
from .export import BUNDLE

_B = {}

# Two resolutions describing the same fix often differ only in their closing
# line -- "Ticket closed", "User confirmed the issue was resolved". Grouping on
# exact text leaves the resolver reading one fix three times, which is the
# problem collapsing was meant to solve. Similarity on the token set is used
# instead of a list of known phrases, so this keeps working on an export whose
# boilerplate is written differently.
SAME_FIX_ABOVE = 0.80


def _fix_key(text):
    return frozenset(w for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2)


def _same_fix(a, b):
    if not a or not b:
        return False
    inter = len(a & b)
    return inter / max(len(a | b), 1) >= SAME_FIX_ABOVE

# Calibrated on the held-out set: at this similarity every flagged pair shared
# the same underlying problem (47 of 924 queries). A duplicate is not a separate
# capability -- it is this model's own similarity scale at its top end, which is
# why it needs no new objective to justify it.
DUPLICATE_ABOVE = 0.85

# How often the retrieved resolution actually transfers -- that is, shares the
# query's underlying problem, so the same fix applies. Measured on the held-out
# set, by similarity band:
#
#     >= 0.80   0.988      0.60-0.70   0.865
#     0.70-0.80 0.960      0.50-0.60   0.641
#
# The bands below name those rates rather than inventing labels. A resolver
# deciding whether to apply someone else's fix needs the rate, not a colour.
APPLICABILITY = [
    (0.80, "very likely to apply", 0.988),
    (0.70, "likely to apply", 0.960),
    (0.60, "may apply", 0.865),
    (0.50, "uncertain", 0.641),
    (0.00, "unlikely to apply", 0.000),
]
# At or above this the interface offers to resolve the ticket with the
# suggestion outright, rather than merely showing it for reference.
RESOLVE_ABOVE = 0.70

# Retrieve more tickets than are shown, so that distinct resolutions can be
# ranked by how much evidence stands behind each. Table 4.19: collapsing the
# top 15 and ordering by summed similarity across the tickets backing a fix
# lifts the chance the first option applies from 0.897 to 0.920, at no cost but
# a wider retrieval. Four distinct options is what that leaves on screen.
GROUP_FROM = 15
SHOW_RESOLUTIONS = 4


def _applicability(sim):
    for floor, label, rate in APPLICABILITY:
        if sim >= floor:
            return {"band": label, "observed_transfer_rate": rate, "floor": floor}
    return {"band": "unlikely to apply", "observed_transfer_rate": 0.0, "floor": 0.0}


def load(path=BUNDLE):
    if path not in _B:
        if not os.path.exists(path):
            raise SystemExit(f"no bundle at {path}\nrun:  make m3-export")
        _B[path] = joblib.load(path)
    return _B[path]


def predict(subject="", description="", category=None, top_k=GROUP_FROM,
            frame=None, bundle=None):
    b = bundle or load()
    f = frame if frame is not None else ticket_frame(subject, description)
    raw = f"{f['Subject'].iloc[0]}. {f['Description'].iloc[0]}".strip()

    q_sem = encode(b["encoder"], [raw])
    s_sem = (q_sem @ b["embeddings"].T)[0]
    s_lex = cosine_similarity(b["lex_vectorizer"].transform(f["text"]),
                              b["lex_matrix"])[0]
    w = b["blend_w"]
    sim = w * s_sem + (1 - w) * s_lex

    scoped = sim
    filtered = False
    if category:
        mask = np.array(b["categories"]) == category
        if mask.sum() >= top_k:
            scoped = np.where(mask, sim, -np.inf)
            filtered = True

    top = np.argsort(-scoped)[:top_k]
    top = [int(i) for i in top if np.isfinite(scoped[i])]
    recs = b["records"]
    thresh = b["strong_match"]
    out = [{
        "ticket_id": recs[i]["Ticket ID"], "subject": recs[i]["Subject"],
        "category": recs[i]["Category"], "team": recs[i]["Assigned Team"],
        "resolution": recs[i]["Resolution Notes"],
        "similarity": round(float(sim[i]), 4),
        "weak": bool(sim[i] < thresh),
    } for i in top]

    dup = out[0] if out and out[0]["similarity"] >= DUPLICATE_ABOVE else None
    # Collapse to DISTINCT resolutions. Of five retrieved tickets only about two
    # carry different answers, and for roughly a quarter of queries all five are
    # the same fix repeated. A resolver wants a short list of options, not one
    # option listed five times; the count of tickets behind each is itself
    # evidence, so it is kept rather than discarded.
    grouped = []
    for m in out:
        key = (m["resolution"] or "").strip()
        fk = _fix_key(key)
        hit = next((g for g in grouped
                    if g["resolution"] == key or _same_fix(g["_fix"], fk)), None)
        if hit:
            hit["seen_in"] += 1
            hit["tickets"].append(m["ticket_id"])
            hit["evidence"] += m["similarity"]
        else:
            grouped.append({
                "resolution": key, "_fix": fk,
                "seen_in": 1, "tickets": [m["ticket_id"]],
                "ticket_id": m["ticket_id"], "subject": m["subject"],
                "category": m["category"], "team": m["team"],
                "similarity": m["similarity"], "weak": m["weak"],
                "evidence": m["similarity"],
                "applicability": _applicability(m["similarity"]),
                "can_resolve": bool(m["similarity"] >= RESOLVE_ABOVE),
            })
    # ordered by the evidence standing behind each fix, not by the single
    # closest ticket that happens to mention it
    grouped.sort(key=lambda g: g["evidence"], reverse=True)
    for g in grouped:
        g["evidence"] = round(g["evidence"], 4)
        g.pop("_fix", None)
    grouped = grouped[:SHOW_RESOLUTIONS]

    return {
        "matches": out[:5],
        "resolutions": grouped,
        "resolve_above": RESOLVE_ABOVE,
        "duplicate_of": dup,
        "duplicate_above": DUPLICATE_ABOVE,
        "strong_match": bool(out and out[0]["similarity"] >= thresh),
        "threshold": thresh,
        "top_similarity": out[0]["similarity"] if out else 0.0,
        "category_filtered": filtered,
        "variant": b["best_variant"],
    }


def card():
    b = load()
    return {
        "name": "Historical-resolution recommendation",
        "thesis": "B3",
        "objective": "Objective 4 / RQ4",
        "head": b["best_variant"],
        "features": f"semantic + lexical blend (w={b['blend_w']}), category-filtered",
        "encoder": b["encoder"],
        "classes": len(b["records"]),
        "metrics": [
            {"label": "Top-1", "value": b.get("top1")},
            {"label": "MRR", "value": b.get("mrr")},
        ],
        "note": ("Retrieval, not classification — it ranks rather than committing, "
                 "which is why it is the most forgiving of the three decisions. "
                 "The one decision where Arm B beats Arm A outright."),
    }
