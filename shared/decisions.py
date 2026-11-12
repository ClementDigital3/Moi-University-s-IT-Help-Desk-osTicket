"""
The human half of the loop.

The models suggest; a person decides. This module records what the person
actually did with each suggestion, which is the point of the artefact: Chapter
Five argues for routing as a ranked suggestion subject to confirmation rather
than an automatic assignment, and that argument only means something if the
confirmation step exists and is measured.

What gets recorded is the pair -- what was suggested, and what was chosen. The
difference between them is the override rate, which is the resolver-facing
evaluation criterion Section 5.4.2 asks future work to use instead of
classification metrics alone.

Appended as JSON Lines so a running demo never rewrites earlier rows, and so
the file stays readable if a run is interrupted.
"""
import os, json, time
from collections import Counter

from .paths import RESULTS

LOG = os.path.join(RESULTS, "decisions.jsonl")


def record(entry):
    """Append one triage decision. Returns the stored row."""
    row = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "arm": entry.get("arm", ""),
        "subject": (entry.get("subject") or "")[:300],
        "description": (entry.get("description") or "")[:1000],
        "suggested_category": entry.get("suggested_category"),
        "chosen_category": entry.get("chosen_category"),
        "suggested_team": entry.get("suggested_team"),
        "chosen_team": entry.get("chosen_team"),
        "resolution_used": entry.get("resolution_used"),
        "resolved_with": entry.get("resolved_with"),
        "referred_to": entry.get("referred_to"),
        "closed_as_duplicate_of": entry.get("closed_as_duplicate_of"),
        "abstained": bool(entry.get("abstained")),
        # A rejection says the suggestion is wrong without naming what is right;
        # an override names it. Keeping them apart matters because a resolver
        # who rejects often does not yet know the answer -- which is what the
        # enrichment step below is for.
        "rejected": bool(entry.get("rejected")),
        "enriched": bool(entry.get("enriched")),
        "text_before": (entry.get("text_before") or "")[:600],
        "text_after": (entry.get("text_after") or "")[:600],
        "confidence_before": entry.get("confidence_before"),
        "confidence_after": entry.get("confidence_after"),
        "confidence_category": entry.get("confidence_category"),
        "confidence_team": entry.get("confidence_team"),
    }
    row["category_overridden"] = bool(
        row["chosen_category"] and row["suggested_category"]
        and row["chosen_category"] != row["suggested_category"])
    row["team_overridden"] = bool(
        row["chosen_team"] and row["suggested_team"]
        and row["chosen_team"] != row["suggested_team"])

    os.makedirs(RESULTS, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def read(limit=None):
    if not os.path.exists(LOG):
        return []
    rows = []
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a partially-written line never breaks the demo
    return rows[-limit:] if limit else rows


def summary():
    """
    Accept/override counts, and where the overrides go.

    Deliberately reports accepted-vs-overridden rather than 'accuracy': the
    resolver's correction is the ground truth here, not a held-out label.
    """
    rows = read()
    n = len(rows)
    cat = [r for r in rows if r.get("chosen_category")]
    team = [r for r in rows if r.get("chosen_team")]
    cat_over = sum(1 for r in cat if r["category_overridden"])
    team_over = sum(1 for r in team if r["team_overridden"])
    flows = Counter((r["suggested_team"], r["chosen_team"])
                    for r in team if r["team_overridden"])
    referrals = [r for r in rows if r.get("referred_to")]
    dups = [r for r in rows if r.get("closed_as_duplicate_of")]
    rejects = [r for r in rows if r.get("rejected")]
    # Enrichment: the resolver supplied the detail the user left out, and the
    # ticket was scored again. The lift is the evidence for Section 5.4.1's
    # recommendation that a better intake form beats a bigger model.
    enriched = [r for r in rows if r.get("enriched")
                and r.get("confidence_before") is not None
                and r.get("confidence_after") is not None]
    lifts = [r["confidence_after"] - r["confidence_before"] for r in enriched]
    return {
        "total": n,
        "category": {"decided": len(cat), "overridden": cat_over,
                     "accept_rate": round(1 - cat_over / len(cat), 3) if cat else None},
        "team": {"decided": len(team), "overridden": team_over,
                 "accept_rate": round(1 - team_over / len(team), 3) if team else None},
        "resolutions_reused": sum(1 for r in rows if r.get("resolution_used")),
        "resolved_from_past": sum(1 for r in rows if r.get("resolved_with")),
        "referred": len(referrals),
        "closed_as_duplicate": len(dups),
        "abstained": sum(1 for r in rows if r.get("abstained")),
        # The share of assignments that did not hold -- the reassignment rate
        # Section 5.4.2 asks future work to evaluate on. The corpus itself
        # carries a Reassigned field showing 13.5% historically, so this is
        # directly comparable.
        "referral_rate": round(len(referrals) / len(team), 3) if team else None,
        "rejected": len(rejects),
        "enriched": len(enriched),
        "mean_confidence_lift": round(sum(lifts) / len(lifts), 3) if lifts else None,
        "enrichment_examples": [
            {"before": r["text_before"][:160], "after": r["text_after"][:160],
             "from": r["confidence_before"], "to": r["confidence_after"]}
            for r in enriched[-3:]][::-1],
        "top_overrides": [{"from": a, "to": b, "n": c}
                          for (a, b), c in flows.most_common(5)],
        "recent": rows[-5:][::-1],
    }
