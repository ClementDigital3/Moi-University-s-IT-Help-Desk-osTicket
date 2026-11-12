"""
Turning resolver decisions into training data.

Section 1.1 describes the help desk record as "an accumulated body of
institutional knowledge: prior problem descriptions paired with the categories,
resolvers and resolutions ultimately applied to them." It accumulates. A system
fitted once to a snapshot of it stops learning the moment it is deployed, which
is at odds with the premise it was built on.

Every decision the artefact records is a labelled example: the ticket text, and
the category and team a human confirmed for it. This module promotes those
decisions into rows the pipeline can train on.

THE HAZARD, AND WHY PROVENANCE IS KEPT
--------------------------------------
A feedback loop that trains on its own accepted suggestions reinforces whatever
the model already believed, errors included, and the more the resolver trusts it
the tighter that loop becomes. Automation bias is precisely the "resolver trust"
risk Section 5.4.2 raises. So a decision is never stored as a bare label; it
carries how it was reached:

    corrected  the resolver overrode, rejected or referred on -- they disagreed
               with the model. This is the signal worth having, because it is
               information the model did not already contain.
    enriched   the resolver supplied detail the ticket lacked. The most valuable
               of all: text with the ambiguity removed, which Chapter Four
               identifies as the ceiling on performance.
    confirmed  the resolver accepted the suggestion. Weak evidence -- it may be
               agreement, or it may be a rubber stamp -- and so it is kept
               separate and excluded by default.

Nothing here decides the right policy. It makes the provenance available so the
policy is an explicit, measurable choice rather than an accident.
"""
import os, json
from collections import Counter

from .paths import ROOT
from .decisions import read as read_decisions

STORE = os.path.join(ROOT, "data", "feedback", "confirmed.jsonl")

CORRECTED, ENRICHED, CONFIRMED = "corrected", "enriched", "confirmed"


def provenance(row):
    """How this decision was reached. Order matters: enrichment outranks the rest."""
    if row.get("enriched"):
        return ENRICHED
    if (row.get("rejected") or row.get("referred_to")
            or row.get("category_overridden") or row.get("team_overridden")):
        return CORRECTED
    if row.get("chosen_category") or row.get("chosen_team"):
        return CONFIRMED
    return None


def to_examples(rows=None):
    """
    Labelled examples from the decision log.

    A decision contributes a label only where a human actually settled one, so a
    rejection with no replacement yields no category or team -- it is recorded
    as evidence the suggestion was wrong, not as a label.
    """
    out = []
    for r in (rows if rows is not None else read_decisions()):
        prov = provenance(r)
        if prov is None:
            continue
        text = (r.get("text_after") or
                f"{r.get('subject','')} {r.get('description','')}").strip()
        if len(text.split()) < 3:
            continue
        out.append({
            "at": r.get("at"), "arm": r.get("arm"),
            "Subject": r.get("subject", ""), "Description": r.get("description", ""),
            "Category": r.get("chosen_category") or "",
            "Assigned Team": r.get("chosen_team") or "",
            "provenance": prov,
            "rejected_only": bool(r.get("rejected") and not r.get("chosen_team")),
            "confidence_before": r.get("confidence_before"),
            "confidence_after": r.get("confidence_after"),
        })
    return out


def promote(rows=None):
    """Append new examples to the store, skipping ones already held."""
    examples = to_examples(rows)
    seen = {(e["at"], e["Subject"], e["Category"], e["Assigned Team"])
            for e in load()}
    fresh = [e for e in examples
             if (e["at"], e["Subject"], e["Category"], e["Assigned Team"]) not in seen]
    if fresh:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        with open(STORE, "a", encoding="utf-8") as f:
            for e in fresh:
                f.write(json.dumps(e) + "\n")
    return fresh


def load():
    if not os.path.exists(STORE):
        return []
    out = []
    with open(STORE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def trainable(policy=("corrected", "enriched")):
    """
    The examples a given policy would train on.

    `confirmed` is absent from the default on purpose -- see the hazard note
    above. Pass it explicitly to include it, and measure what it does.
    """
    return [e for e in load()
            if e["provenance"] in policy
            and e["Category"] and e["Assigned Team"]
            and not e["rejected_only"]]


def summary():
    held = load()
    counts = Counter(e["provenance"] for e in held)
    usable = trainable()
    lifts = [e["confidence_after"] - e["confidence_before"] for e in held
             if e.get("confidence_before") is not None
             and e.get("confidence_after") is not None]
    return {
        "held": len(held),
        "by_provenance": dict(counts),
        "trainable_default_policy": len(usable),
        "rejections_without_a_label": sum(1 for e in held if e["rejected_only"]),
        "mean_enrichment_lift": round(sum(lifts) / len(lifts), 3) if lifts else None,
    }
