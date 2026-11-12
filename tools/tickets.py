"""
The ticket queue, on the terminal.

The queue page answers this in a browser. This answers it without one, which
matters for a quick check over SSH, for pasting a routing table into the thesis,
and for confirming the database holds what the interface claims it holds.

    python -m tools.tickets                 where the tickets are
    python -m tools.tickets MU-0007         one ticket's full history
    python -m tools.tickets --team "Hardware Support"
"""
import argparse

from shared import db

# The corpus records 13.5% of tickets as reassigned. Section 5.4.2 asks future
# work to evaluate routing on reassignment rather than classification accuracy,
# so the live figure is printed against that baseline to be comparable.
CORPUS_REASSIGNMENT = 0.135


def show_journey(ref):
    found = db.journey(ref)
    if found is None:
        print(f"  no ticket {ref}")
        return
    t = found["ticket"]
    print(f"\n  {t['ref']}  {t['subject'] or '(no subject)'}")
    print(f"  {t['status']}  ·  {t['team'] or 'unassigned'}"
          f"  ·  {t['hops']} hand-off{'' if t['hops'] == 1 else 's'}")
    if t["description"]:
        print(f"  {t['description'][:110]}")
    print()
    for e in found["events"]:
        if e["kind"] == "referred":
            where = f"{e['from_team']} -> {e['to_team']}"
        else:
            where = e["to_team"] or e["category"] or (e["detail"] or "")[:60] or ""
        print(f"    {e['at'][11:]}  {e['kind']:10s} {where}")
    print()


def main():
    ap = argparse.ArgumentParser(description="where the tickets went")
    ap.add_argument("ref", nargs="?", help="a ticket reference, e.g. MU-0007")
    ap.add_argument("--team", help="only tickets held by this team")
    ap.add_argument("--status", help="open, referred, resolved or duplicate")
    args = ap.parse_args()

    if args.ref:
        return show_journey(args.ref)

    s = db.stats()
    if not s["total"]:
        print("\n  No tickets yet.\n\n  Accept, refer or resolve a suggestion on any"
              " model page and it lands here.\n")
        return

    print(f"\n  {s['total']} tickets  ·  {s['open']} open  ·  {s['resolved']} resolved"
          f"  ·  {s['duplicate']} duplicate  ·  {s['teams']} teams")
    if s["reassignment_rate"] is not None:
        live = s["reassignment_rate"]
        direction = ("above" if live > CORPUS_REASSIGNMENT else
                     "below" if live < CORPUS_REASSIGNMENT else "level with")
        print(f"  reassignment {live:.1%} — {direction} the corpus baseline of"
              f" {CORPUS_REASSIGNMENT:.1%}"
              f"  ({s['ever_referred']} of {s['total']} handed on)")

    print(f"\n  {'TEAM':30s}{'OPEN':>6}{'TOTAL':>7}{'INHERITED':>11}{'PASSED ON':>11}")
    for r in db.routing():
        print(f"  {r['team'][:29]:30s}{r['open']:>6}{r['total']:>7}"
              f"{r['inherited'] or '-':>11}{r['passed_on'] or '-':>11}")

    hand = db.handoffs()
    if hand:
        print("\n  HAND-OFFS")
        for h in hand:
            print(f"    {h['from_team']} -> {h['to_team']}  ({h['n']})")

    rows = db.queue(status=args.status, team=args.team, limit=20)
    if rows:
        print(f"\n  {'REF':10s}{'STATUS':11s}{'TEAM':28s}SUBJECT")
        for t in rows:
            hops = f" +{t['hops']}" if t["hops"] else ""
            print(f"  {t['ref']:10s}{t['status']:11s}"
                  f"{((t['team'] or '-')[:25] + hops):28s}{(t['subject'] or '')[:40]}")
    print()


if __name__ == "__main__":
    main()
