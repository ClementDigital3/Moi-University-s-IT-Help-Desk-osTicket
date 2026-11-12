"""
Where the tickets went.

The decision log (shared/decisions.py) records one row per click: what was
suggested, what was chosen. That answers "how often do resolvers override the
model", which is what Chapter Five argues for, and it is kept.

What it cannot answer is where a ticket ended up. Every row in that file is
independent -- there is no ticket identity anywhere in it. A ticket that is
suggested, rejected, enriched, re-scored and then referred produces five
unrelated lines, and nothing joins them. Asking "which team is holding this
ticket now" or "how many tickets reached the SIS team by referral rather than
directly" is not a hard query against that file; it is an impossible one,
because the information was never written down.

So: two tables.

  tickets   one row per ticket -- its CURRENT state. Where it is routed.
  events    one row per thing that happened to it, in order. How it got there.

Current state could be derived by replaying events, and purists would do that.
It is stored instead because the question this exists to answer -- where are
the tickets -- should be one indexed read, not a fold over history, and because
a demonstration that has to replay a log to draw a queue is a demonstration
that stalls in front of an examiner.

SQLite because it fits the shape of this project: the standard library already
has it, there is no server to install or credentials to configure, it is a
single file that clones with the repository, and an examiner running this on
another laptop needs nothing beyond `make setup`. A thesis prototype that
demands Postgres before it will show a queue has chosen its own convenience
over its reader's.

The reassignment rate matters here beyond bookkeeping. The corpus carries a
Reassigned field showing 13.5% historically, and Section 5.4.2 asks future work
to evaluate routing on reassignment rather than on classification accuracy.
A referral recorded here is that measurement, taken live.
"""
import os
import sqlite3
import time

from .paths import RESULTS

DB = os.path.join(RESULTS, "helpdesk.db")

# Ticket life cycle. A ticket is in exactly one of these at any moment.
OPEN, REFERRED, RESOLVED, DUPLICATE = "open", "referred", "resolved", "duplicate"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ref           TEXT    UNIQUE,
    opened_at     TEXT    NOT NULL,
    subject       TEXT    NOT NULL DEFAULT '',
    description   TEXT    NOT NULL DEFAULT '',
    arm           TEXT,              -- which arm triaged it: A, B, or a single model
    category      TEXT,              -- settled category (suggested, unless overridden)
    team          TEXT,              -- WHERE IT IS ROUTED. the question this file exists for
    status        TEXT    NOT NULL DEFAULT 'open',
    opened_by     TEXT,
    closed_at     TEXT,
    resolution    TEXT,
    duplicate_of  INTEGER REFERENCES tickets(id),
    hops          INTEGER NOT NULL DEFAULT 0   -- referrals so far; 0 means first team held it
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   INTEGER NOT NULL REFERENCES tickets(id),
    at          TEXT    NOT NULL,
    kind        TEXT    NOT NULL,   -- opened|suggested|accepted|rejected|enriched|
                                    -- referred|resolved|duplicate|reopened
    actor       TEXT,
    from_team   TEXT,
    to_team     TEXT,
    category    TEXT,
    confidence  REAL,
    detail      TEXT
);

CREATE INDEX IF NOT EXISTS ix_tickets_team   ON tickets(team);
CREATE INDEX IF NOT EXISTS ix_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS ix_events_ticket  ON events(ticket_id, id);
"""


def connect():
    """
    A connection with the schema already applied.

    One connection per call rather than a shared handle: the server is a
    ThreadingHTTPServer, and a SQLite connection may not be used from a thread
    other than the one that made it. Opening per call is the honest fix.
    Connecting is microseconds against a local file, and it cannot produce the
    intermittent cross-thread failure that a cached handle eventually will.

    WAL lets a reader (the queue page) and a writer (someone clicking Accept)
    proceed at the same time instead of one meeting 'database is locked'.
    """
    os.makedirs(RESULTS, exist_ok=True)
    con = sqlite3.connect(DB, timeout=5.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    return con


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _event(con, ticket_id, kind, **kw):
    con.execute(
        "INSERT INTO events (ticket_id, at, kind, actor, from_team, to_team,"
        " category, confidence, detail) VALUES (?,?,?,?,?,?,?,?,?)",
        (ticket_id, _now(), kind, kw.get("actor"), kw.get("from_team"),
         kw.get("to_team"), kw.get("category"), kw.get("confidence"),
         kw.get("detail")))


def open_ticket(subject="", description="", arm=None, category=None,
                team=None, confidence=None, actor=None):
    """
    Register a ticket and the routing it was given. Returns its row.

    Called when a suggestion is acted on, not when it is merely displayed:
    scoring a ticket to look at the output is not the same as a ticket
    entering the queue, and counting the former as the latter would inflate
    every rate computed from this table.
    """
    con = connect()
    try:
        cur = con.execute(
            "INSERT INTO tickets (ref, opened_at, subject, description, arm,"
            " category, team, status, opened_by) VALUES (?,?,?,?,?,?,?,?,?)",
            (None, _now(), (subject or "")[:300], (description or "")[:2000],
             arm, category, team, OPEN, actor))
        tid = cur.lastrowid
        # Human-readable reference, assigned after the id is known so the two
        # never disagree. MU for Moi University.
        con.execute("UPDATE tickets SET ref=? WHERE id=?", (f"MU-{tid:04d}", tid))
        _event(con, tid, "opened", actor=actor, to_team=team,
               category=category, confidence=confidence)
        con.commit()
        return dict(con.execute("SELECT * FROM tickets WHERE id=?", (tid,)).fetchone())
    finally:
        con.close()


def refer(ticket_id, to_team, actor=None, detail=None):
    """
    Hand a ticket to another team, and remember that it moved.

    The hop count is what makes this worth storing. A ticket that reaches the
    right team after three hand-offs was routed badly even though its final
    team is correct, and an accuracy figure computed on the final team alone
    would score it as a success. Chapter Five's argument is precisely that this
    is the wrong measurement.
    """
    con = connect()
    try:
        row = con.execute("SELECT team FROM tickets WHERE id=?", (ticket_id,)).fetchone()
        if row is None:
            return None
        frm = row["team"]
        # Referring a ticket to the team already holding it is not a hand-off.
        # Counting it as one would let the reassignment rate -- a figure this
        # project reports against the corpus's 13.5% -- be inflated by a
        # no-op, so the move is refused rather than recorded.
        if frm is not None and frm == to_team:
            return dict(con.execute("SELECT * FROM tickets WHERE id=?",
                                    (ticket_id,)).fetchone())
        con.execute("UPDATE tickets SET team=?, status=?, hops=hops+1 WHERE id=?",
                    (to_team, REFERRED, ticket_id))
        _event(con, ticket_id, "referred", actor=actor,
               from_team=frm, to_team=to_team, detail=detail)
        con.commit()
        return dict(con.execute("SELECT * FROM tickets WHERE id=?",
                                (ticket_id,)).fetchone())
    finally:
        con.close()


def resolve(ticket_id, resolution=None, actor=None):
    """Close a ticket, recording what fixed it."""
    con = connect()
    try:
        con.execute("UPDATE tickets SET status=?, closed_at=?, resolution=? WHERE id=?",
                    (RESOLVED, _now(), (resolution or "")[:2000] or None, ticket_id))
        _event(con, ticket_id, "resolved", actor=actor, detail=(resolution or "")[:500])
        con.commit()
        return dict(con.execute("SELECT * FROM tickets WHERE id=?",
                                (ticket_id,)).fetchone())
    finally:
        con.close()


def mark_duplicate(ticket_id, of_ref_or_id, actor=None):
    """Close a ticket as already-reported, pointing at the original."""
    con = connect()
    try:
        con.execute("UPDATE tickets SET status=?, closed_at=?, duplicate_of=? WHERE id=?",
                    (DUPLICATE, _now(), _resolve_ref(con, of_ref_or_id), ticket_id))
        _event(con, ticket_id, "duplicate", actor=actor, detail=str(of_ref_or_id))
        con.commit()
        return dict(con.execute("SELECT * FROM tickets WHERE id=?",
                                (ticket_id,)).fetchone())
    finally:
        con.close()


def _resolve_ref(con, value):
    """Accept either a numeric id or an MU-0001 style reference."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    row = con.execute("SELECT id FROM tickets WHERE ref=?", (text,)).fetchone()
    return row["id"] if row else None


def note(ticket_id, kind, **kw):
    """Record something that happened without changing where the ticket sits."""
    con = connect()
    try:
        _event(con, ticket_id, kind, **kw)
        con.commit()
    finally:
        con.close()


def journey(ticket_id):
    """Everything that happened to one ticket, oldest first."""
    con = connect()
    try:
        tid = _resolve_ref(con, ticket_id)
        t = con.execute("SELECT * FROM tickets WHERE id=?", (tid,)).fetchone()
        if t is None:
            return None
        evs = con.execute("SELECT * FROM events WHERE ticket_id=? ORDER BY id",
                          (tid,)).fetchall()
        return {"ticket": dict(t), "events": [dict(e) for e in evs]}
    finally:
        con.close()


def queue(status=None, team=None, limit=100):
    """Tickets, most recent first, optionally narrowed to a status or team."""
    con = connect()
    try:
        sql = "SELECT * FROM tickets"
        where, args = [], []
        if status:
            where.append("status=?"); args.append(status)
        if team:
            where.append("team=?"); args.append(team)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


def routing():
    """
    Where the tickets are. The question this module exists to answer.

    Reports per team: how many tickets sit with it in total, how many of those
    are still open, how many arrived by referral rather than directly, and how
    many it has since handed on. Sorted by open work first, because that is the
    column a help desk manager acts on. A team with a high inherited count is absorbing other teams'
    misroutes; a high passed-on count is the opposite. Neither is visible in a
    confusion matrix, and both are what a help desk manager would ask about.
    """
    con = connect()
    try:
        teams = {}
        for r in con.execute(
                "SELECT team, status, COUNT(*) n FROM tickets"
                " WHERE team IS NOT NULL GROUP BY team, status"):
            t = teams.setdefault(r["team"], {"team": r["team"], "total": 0,
                                             "open": 0, "resolved": 0,
                                             "inherited": 0, "passed_on": 0})
            t["total"] += r["n"]
            if r["status"] in (OPEN, REFERRED):
                t["open"] += r["n"]
            if r["status"] == RESOLVED:
                t["resolved"] += r["n"]
        for r in con.execute(
                "SELECT to_team, COUNT(*) n FROM events WHERE kind='referred'"
                " AND to_team IS NOT NULL GROUP BY to_team"):
            teams.setdefault(r["to_team"], {"team": r["to_team"], "total": 0,
                                            "open": 0, "resolved": 0,
                                            "inherited": 0, "passed_on": 0})
            teams[r["to_team"]]["inherited"] = r["n"]
        for r in con.execute(
                "SELECT from_team, COUNT(*) n FROM events WHERE kind='referred'"
                " AND from_team IS NOT NULL GROUP BY from_team"):
            teams.setdefault(r["from_team"], {"team": r["from_team"], "total": 0,
                                              "open": 0, "resolved": 0,
                                              "inherited": 0, "passed_on": 0})
            teams[r["from_team"]]["passed_on"] = r["n"]
        return sorted(teams.values(), key=lambda t: (-t["open"], -t["total"]))
    finally:
        con.close()


def handoffs(limit=10):
    """The most common team-to-team hand-offs: which misroutes actually happen."""
    con = connect()
    try:
        return [dict(r) for r in con.execute(
            "SELECT from_team, to_team, COUNT(*) n FROM events"
            " WHERE kind='referred' AND from_team IS NOT NULL"
            " GROUP BY from_team, to_team ORDER BY n DESC LIMIT ?", (limit,))]
    finally:
        con.close()


def stats():
    """Headline counts for the queue page and the CLI."""
    con = connect()
    try:
        one = lambda q, *a: con.execute(q, a).fetchone()[0]
        total = one("SELECT COUNT(*) FROM tickets")
        referred = one("SELECT COUNT(*) FROM tickets WHERE hops > 0")
        return {
            "total": total,
            "open": one("SELECT COUNT(*) FROM tickets WHERE status IN (?,?)",
                        OPEN, REFERRED),
            "resolved": one("SELECT COUNT(*) FROM tickets WHERE status=?", RESOLVED),
            "duplicate": one("SELECT COUNT(*) FROM tickets WHERE status=?", DUPLICATE),
            "ever_referred": referred,
            # Directly comparable with the corpus's historical 13.5%, which is
            # the point of measuring it the same way.
            "reassignment_rate": round(referred / total, 3) if total else None,
            "teams": one("SELECT COUNT(DISTINCT team) FROM tickets WHERE team IS NOT NULL"),
        }
    finally:
        con.close()


def apply_decision(entry):
    """
    Translate one decision from the UI into ticket state.

    The pages already post what happened -- a chosen team, a referral, a
    resolution, a duplicate. This turns that into a ticket that exists
    somewhere, so the same click feeds both the override statistics and the
    queue.

    `db_ref` is how a sequence stays one ticket. A resolver who rejects a
    suggestion, enriches the ticket and then refers it has performed three
    actions on ONE ticket; without the reference coming back each time, that
    would open three, and every per-ticket figure computed afterwards -- hop
    counts, reassignment rate, how many tickets a team actually saw -- would be
    inflated by the interface rather than by the help desk.
    """
    con = connect()
    try:
        tid = _resolve_ref(con, entry.get("db_ref"))
        if tid is not None:
            exists = con.execute("SELECT id FROM tickets WHERE id=?", (tid,)).fetchone()
            if exists is None:
                tid = None
    finally:
        con.close()

    category = entry.get("chosen_category") or entry.get("suggested_category")
    team = entry.get("chosen_team") or entry.get("suggested_team")
    actor = entry.get("actor") or "resolver"

    if tid is None:
        row = open_ticket(subject=entry.get("subject", ""),
                          description=entry.get("description", ""),
                          arm=entry.get("arm"), category=category, team=team,
                          confidence=entry.get("confidence_team"), actor=actor)
        tid = row["id"]
    else:
        # An existing ticket may have had its category or team corrected.
        con = connect()
        try:
            if category or team:
                con.execute("UPDATE tickets SET category=COALESCE(?, category),"
                            " team=COALESCE(?, team) WHERE id=?",
                            (category, team, tid))
                con.commit()
        finally:
            con.close()

    if entry.get("rejected"):
        note(tid, "rejected", actor=actor, category=entry.get("suggested_category"),
             confidence=entry.get("confidence_team"))
    if entry.get("enriched"):
        note(tid, "enriched", actor=actor,
             confidence=entry.get("confidence_after"),
             detail=(entry.get("text_after") or "")[:500])
    if entry.get("category_overridden") or entry.get("team_overridden"):
        note(tid, "accepted" if not entry.get("rejected") else "corrected",
             actor=actor, from_team=entry.get("suggested_team"),
             to_team=entry.get("chosen_team"), category=category)

    if entry.get("referred_to"):
        refer(tid, entry["referred_to"], actor=actor, detail="referred from the queue")
    elif entry.get("closed_as_duplicate_of"):
        mark_duplicate(tid, entry["closed_as_duplicate_of"], actor=actor)
    elif entry.get("resolved_with"):
        resolve(tid, f"reused the fix from {entry['resolved_with']}", actor=actor)

    con = connect()
    try:
        row = con.execute("SELECT * FROM tickets WHERE id=?", (tid,)).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def reopen(ticket_id, actor=None, detail=None):
    """
    Put a closed ticket back into the queue.

    A resolution that did not hold is the most informative thing a help desk
    records: it says the suggested fix was applied and was wrong, which is a
    stronger negative signal than a resolver declining the suggestion in the
    first place. Closing it again later would otherwise erase that.
    """
    con = connect()
    try:
        row = con.execute("SELECT status FROM tickets WHERE id=?",
                          (ticket_id,)).fetchone()
        if row is None:
            return None
        con.execute("UPDATE tickets SET status=?, closed_at=NULL WHERE id=?",
                    (OPEN, ticket_id))
        _event(con, ticket_id, "reopened", actor=actor, detail=detail)
        con.commit()
        return dict(con.execute("SELECT * FROM tickets WHERE id=?",
                                (ticket_id,)).fetchone())
    finally:
        con.close()


def assign(ticket_id, team, actor=None):
    """
    Set the holding team without counting it as a hand-off.

    Distinct from refer() on purpose. Correcting a ticket that was never
    actioned is not the same event as a team working it and passing it on, and
    folding the two together would inflate the reassignment rate with
    corrections that no resolver ever had to absorb.
    """
    con = connect()
    try:
        row = con.execute("SELECT team FROM tickets WHERE id=?",
                          (ticket_id,)).fetchone()
        if row is None:
            return None
        con.execute("UPDATE tickets SET team=? WHERE id=?", (team, ticket_id))
        _event(con, ticket_id, "assigned", actor=actor,
               from_team=row["team"], to_team=team)
        con.commit()
        return dict(con.execute("SELECT * FROM tickets WHERE id=?",
                                (ticket_id,)).fetchone())
    finally:
        con.close()


def teams():
    """Every team seen so far, for populating a referral menu."""
    con = connect()
    try:
        return [r["team"] for r in con.execute(
            "SELECT DISTINCT team FROM tickets WHERE team IS NOT NULL"
            " UNION SELECT DISTINCT to_team FROM events"
            " WHERE to_team IS NOT NULL ORDER BY 1")]
    finally:
        con.close()
