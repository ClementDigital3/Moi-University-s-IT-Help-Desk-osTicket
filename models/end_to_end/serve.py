"""
Local web prototype for the end-to-end model.

A single self-contained page served from the Python standard library. No web
framework, no CDN, no network access -- it runs on a laptop with the wifi off,
which is the only assumption worth making about a presentation room.

    .venv/bin/python -m models.end_to_end.serve
    .venv/bin/python -m models.end_to_end.serve --port 8080 --open
    .venv/bin/python -m models.end_to_end.serve --lan

--lan binds every interface instead of loopback, so other devices on the same
network can reach it -- the machine driving the projector, a phone, a
supervisor's laptop in the room.

--auth puts HTTP Basic authentication in front of everything. Give it
user:password, or pass it bare to have a password generated and printed once.
Loopback is exempt, so working on this machine is unaffected.

Anything reachable beyond the local network should use it. The prototype serves
unpublished Chapter Four findings and runs model inference on whatever it is
sent; neither belongs on an open URL. There is no account system here and none
is wanted -- one shared credential, held for as long as the demonstration
lasts.

The bundle is loaded once at startup, so the first prediction is as fast as the
hundredth. Export it first with models.end_to_end.export.
"""
import os, json, argparse, webbrowser, base64, hmac, secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from shared.paths import ROOT
from .predict import load, predict

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "app.html")
CASCADE_PAGE = os.path.join(ROOT, "models", "cascade.html")
TICKETS_PAGE = os.path.join(ROOT, "models", "tickets.html")
SUBMIT_PAGE = os.path.join(ROOT, "models", "submit.html")
MODEL_PAGE = os.path.join(ROOT, "models", "model.html")
# One stylesheet for every page, so colour, type and spacing cannot drift
# between them. The dashboard inlines the same file at build time, because it
# must also work as a standalone file with no server.
UI_CSS = os.path.join(ROOT, "shared", "ui.css")
UI_JS = os.path.join(ROOT, "shared", "ui.js")

# Each Arm B model gets its own page. They share one shell, parameterised by
# name, because the pages differ in what they show -- not in how they are built.
MODEL_ROUTES = {"/classification": "classification",
                "/routing": "routing",
                "/recommendation": "recommendation"}
# The evidence dashboard is a standalone file, but serving it from here too
# means the two views are one browsable prototype: no alt-tabbing to a file
# manager mid-presentation. Read fresh per request so a rebuild shows up
# without restarting the server.
DASHBOARD = os.path.join(ROOT, "results", "dashboard.html")

DASH_MISSING = """<!doctype html><meta charset="utf-8">
<title>Dashboard not built</title>
<body style="font:16px/1.6 system-ui;max-width:44rem;margin:12vh auto;padding:0 1.5rem">
<h1 style="font-size:1.4rem">The evidence dashboard has not been built yet</h1>
<p>Build it, then reload this page &mdash; the server picks it up without restarting:</p>
<pre style="background:#f1f3f8;padding:.9rem 1rem;border-radius:8px;overflow:auto"
>.venv/bin/python -m analysis.build_dashboard</pre>
<p><a href="/">&larr; Back to live triage</a></p></body>"""

EXAMPLES = [
    {"label": "Ambiguous login",
     "subject": "Cannot login",
     "description": "I have been trying since morning and it keeps failing. Please help."},
    {"label": "LMS upload",
     "subject": "Cannot upload course materials",
     "description": "The e-learning platform rejects my lecture notes when I try to "
                    "attach them to the course page. It says the file is too large."},
    {"label": "Wi-Fi in a lecture hall",
     "subject": "No internet in the lecture hall",
     "description": "The wifi shows connected but no pages load. This is in the "
                    "engineering block, affecting the whole class."},
    {"label": "Transcript / records",
     "subject": "Missing unit in my transcript",
     "description": "Two units I completed last semester are not showing on my "
                    "student records portal transcript."},
    {"label": "Broken hardware",
     "subject": "Printer not working",
     "description": "The office printer makes a grinding noise and jams on every "
                    "page. We have tried switching it off and on."},
]


AUTH = {"user": None, "password": None}


def _authorised(handler):
    """
    True if this request may proceed.

    Loopback is exempt: a password on 127.0.0.1 only gets in the way of the
    person running the server. Everything arriving over a network interface is
    checked. Comparison is constant-time so a wrong password cannot be found a
    character at a time.
    """
    if not AUTH["password"]:
        return True
    if handler.client_address[0] in ("127.0.0.1", "::1"):
        return True
    header = handler.headers.get("Authorization", "")
    if not header.startswith("Basic "):
        return False
    try:
        got = base64.b64decode(header[6:]).decode("utf-8", "replace")
    except Exception:
        return False
    want = f"{AUTH['user']}:{AUTH['password']}"
    return hmac.compare_digest(got, want)


class Handler(BaseHTTPRequestHandler):
    def _challenge(self):
        body = b"Authentication required."
        self.send_response(401)
        self.send_header("WWW-Authenticate",
                         'Basic realm="ICT Help Desk prototype", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not _authorised(self):
            return self._challenge()
        # The page accepts ?subject=&description= for bookmarkable demo links,
        # so route on the path alone.
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            with open(PAGE, "r", encoding="utf-8") as f:
                html = f.read()
            # A prefilled link is scored server-side and injected, so the page
            # arrives with its answer already on it -- no loading flash in front
            # of an audience, and the link is shareable as a fixed example.
            q = parse_qs(parsed.query)
            subj = (q.get("subject") or [""])[0]
            desc = (q.get("description") or [""])[0]
            if subj or desc:
                try:
                    preload = json.dumps(predict(subj, desc))
                except Exception as e:
                    preload = json.dumps({"error": f"{type(e).__name__}: {e}"})
                html = html.replace(
                    "</head>",
                    f"<script>window.__PRELOAD__={preload};</script></head>", 1)
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if path in ("/dashboard", "/dashboard.html"):
            if not os.path.exists(DASHBOARD):
                return self._send(200, DASH_MISSING, "text/html; charset=utf-8")
            with open(DASHBOARD, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path in ("/cascade", "/cascade.html"):
            with open(CASCADE_PAGE, "r", encoding="utf-8") as f:
                html = f.read()
            q = parse_qs(parsed.query)
            subj = (q.get("subject") or [""])[0]
            desc = (q.get("description") or [""])[0]
            if subj or desc:
                try:
                    pre = json.dumps(cascade(subj, desc))
                except Exception as e:
                    pre = json.dumps({"error": f"{type(e).__name__}: {e}"})
                html = html.replace(
                    "</head>", f"<script>window.__PRELOAD__={pre};</script></head>", 1)
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/static/ui.css":
            with open(UI_CSS, "rb") as f:
                return self._send(200, f.read(), "text/css; charset=utf-8")
        if path == "/static/ui.js":
            with open(UI_JS, "rb") as f:
                return self._send(200, f.read(), "application/javascript; charset=utf-8")
        if path in MODEL_ROUTES:
            name = MODEL_ROUTES[path]
            with open(MODEL_PAGE, "r", encoding="utf-8") as f:
                html = f.read()
            inject = f"window.__MODEL__={json.dumps(name)};"
            q = parse_qs(parsed.query)
            subj = (q.get("subject") or [""])[0]
            desc = (q.get("description") or [""])[0]
            if subj or desc:
                try:
                    inject += f"window.__PRELOAD__={json.dumps(run_model(name, subj, desc))};"
                except Exception as e:
                    inject += f'window.__PRELOAD__={json.dumps({"error": str(e)})};'
            html = html.replace("</head>", f"<script>{inject}</script></head>", 1)
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/api/decisions":
            from shared.decisions import summary
            return self._send(200, json.dumps(summary()))
        if path.startswith("/api/card/"):
            name = path.rsplit("/", 1)[-1]
            if name not in MODEL_ROUTES.values():
                return self._send(404, json.dumps({"error": "unknown model"}))
            try:
                return self._send(200, json.dumps(_card(name)))
            except SystemExit as e:
                return self._send(200, json.dumps({"error": str(e)}))
        if path == "/api/examples":
            return self._send(200, json.dumps(EXAMPLES))
        if path in ("/submit", "/submit.html"):
            with open(SUBMIT_PAGE, "r", encoding="utf-8") as f:
                return self._send(200, f.read().encode("utf-8"),
                                  "text/html; charset=utf-8")
        if path == "/api/teams":
            from shared import db
            # Every team the router knows, not merely the ones seen so far: a
            # menu built only from past tickets cannot offer the team a ticket
            # has never reached, which is exactly the misroute worth fixing.
            seen = set(db.teams())
            try:
                from models.routing.predict import load as load_routing
                seen |= set(load_routing()["classes"])
            except Exception:
                pass                     # a referral menu is worth having anyway
            return self._send(200, json.dumps(sorted(t for t in seen if t)))
        if path in ("/tickets", "/tickets.html"):
            with open(TICKETS_PAGE, "r", encoding="utf-8") as f:
                return self._send(200, f.read().encode("utf-8"),
                                  "text/html; charset=utf-8")
        if path == "/api/tickets":
            from shared import db
            q = parse_qs(parsed.query)
            ref = (q.get("ref") or [""])[0]
            if ref:
                found = db.journey(ref)
                if found is None:
                    return self._send(404, json.dumps({"error": "no such ticket"}))
                return self._send(200, json.dumps(found))
            return self._send(200, json.dumps({
                "stats": db.stats(), "routing": db.routing(),
                "handoffs": db.handoffs(),
                "queue": db.queue(status=(q.get("status") or [""])[0] or None,
                                  team=(q.get("team") or [""])[0] or None)}))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if not _authorised(self):
            return self._challenge()
        path = urlparse(self.path).path
        model_api = path.rsplit("/", 1)[-1] if path.startswith("/api/model/") else None
        if path not in ("/api/predict", "/api/cascade", "/api/decision",
                        "/api/intake", "/api/ticket") and \
                model_api not in MODEL_ROUTES.values():
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            subj, desc = payload.get("subject", ""), payload.get("description", "")
            if path == "/api/decision":
                from shared.decisions import record, summary
                from shared import db
                row = record({**payload, "subject": subj, "description": desc})
                # The log keeps the suggested-vs-chosen pair; the database keeps
                # the ticket. Same click, two questions: how often the model is
                # overridden, and where the ticket actually went.
                ticket = db.apply_decision({**payload, **row,
                                            "db_ref": payload.get("db_ref")})
                return self._send(200, json.dumps({
                    "ok": True, "summary": summary(),
                    "ticket": ticket}))
            if path == "/api/intake":
                return self._send(200, json.dumps(intake(payload)))
            if path == "/api/ticket":
                return self._send(200, json.dumps(ticket_action(payload)))
            if path == "/api/predict":
                return self._send(200, json.dumps(predict(subj, desc)))
            if model_api:
                return self._send(200, json.dumps(run_model(model_api, subj, desc)))
            return self._send(200, json.dumps(cascade(subj, desc)))
        except SystemExit as e:                     # a bundle has not been exported
            self._send(200, json.dumps({"error": str(e)}))
        except Exception as e:                      # keep the demo alive
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


# Used only if the routing bundle somehow omits it; the model normally reports
# its own first line, and that value is the one that should win.
FIRST_LINE_FALLBACK = "Help Desk (Tier 1)"


def intake(payload):
    """
    A ticket arrives, and is triaged on the way in.

    This is the front door of the working system: the requester describes a
    problem and the ticket is scored, routed and queued in one step, rather
    than someone pasting text into a model page to see what it says. The
    suggestion is recorded as a suggestion -- the ticket opens with the team
    the model proposed, and the event trail says the model proposed it, so a
    later correction is visible as a correction rather than silently replacing
    history.

    Nothing is auto-resolved. Chapter Five argues for routing as a ranked
    suggestion subject to confirmation, and an intake that closed its own
    tickets would contradict the thesis it is meant to demonstrate.
    """
    from shared import db

    subject = (payload.get("subject") or "").strip()
    description = (payload.get("description") or "").strip()
    if not subject and not description:
        return {"error": "Describe the problem before submitting."}

    try:
        scored = cascade(subject, description)
    except Exception as e:                       # triage must not lose a ticket
        scored = {"error": f"{type(e).__name__}: {e}"}

    category = team = None
    confidence = None
    abstained = False
    proposed = None
    if "error" not in scored:
        r = scored["routing"]
        category = scored["category"]["label"]
        proposed = r["label"]
        confidence = r.get("confidence")
        abstained = bool(r.get("abstain"))
        # An abstention is the model declining to choose. Filing the ticket
        # with its low-confidence guess anyway would make the threshold
        # decorative: the ticket would land on a specialist team on the
        # strength of a score the model itself called too weak to act on.
        # Below the threshold it goes to the first line, which is what the
        # abstention is FOR, and the guess is kept in the trail rather than
        # thrown away.
        team = r.get("first_line", FIRST_LINE_FALLBACK) if abstained else proposed

    row = db.open_ticket(subject=subject, description=description, arm="B",
                         category=category, team=team, confidence=confidence,
                         actor=(payload.get("requester") or "requester"))
    db.note(row["id"], "suggested", actor="model", to_team=team,
            category=category, confidence=confidence,
            detail=(f"abstained at {confidence} -- would have said {proposed};"
                    f" sent to the first line instead") if abstained else None)
    return {"ok": True, "ticket": row, "triage": scored}


def ticket_action(payload):
    """
    Work a ticket that is already in the queue: resolve, refer, reopen, assign.

    Separate from /api/decision, which exists to record a suggestion being
    accepted or overridden at the moment it is shown. This is someone opening
    the queue later and acting on what is there, which is a different event
    with a different meaning for the override statistics -- conflating them
    would count queue housekeeping as model feedback.
    """
    from shared import db

    ref = payload.get("ref")
    action = payload.get("action")
    actor = payload.get("actor") or "resolver"
    found = db.journey(ref)
    if found is None:
        return {"error": "no such ticket"}
    tid = found["ticket"]["id"]

    if action == "resolve":
        row = db.resolve(tid, payload.get("resolution"), actor=actor)
    elif action == "refer":
        to = (payload.get("team") or "").strip()
        if not to:
            return {"error": "Choose a team to refer to."}
        row = db.refer(tid, to, actor=actor, detail=payload.get("note"))
    elif action == "reopen":
        row = db.reopen(tid, actor=actor, detail=payload.get("note"))
    elif action == "assign":
        to = (payload.get("team") or "").strip()
        if not to:
            return {"error": "Choose a team."}
        row = db.assign(tid, to, actor=actor)
    elif action == "duplicate":
        row = db.mark_duplicate(tid, payload.get("of"), actor=actor)
    else:
        return {"error": f"unknown action: {action}"}
    return {"ok": True, "ticket": row}


def _card(name):
    if name == "classification":
        from models.classification.predict import card
    elif name == "routing":
        from models.routing.predict import card
    else:
        from models.recommendation.predict import card
    return card()


def run_model(name, subject, description):
    """
    Score one ticket with ONE of the Arm B models.

    Routing and recommendation are cascaded, so running them alone still means
    running classification first. The upstream result is returned alongside
    rather than hidden, because on a page about one model the dependency is
    exactly what a reader needs to see.
    """
    from shared.serving import ticket_frame
    from models.classification.predict import predict as p1

    if not (subject or "").strip() and not (description or "").strip():
        return {"error": "Enter a subject or a description."}

    frame = ticket_frame(subject, description)
    cat = p1(frame=frame)
    vec = cat.pop("proba_vector")

    if name == "classification":
        return {"model": name, "result": cat}
    if name == "routing":
        from models.routing.predict import predict as p2
        return {"model": name, "upstream": {"label": cat["label"],
                                            "confidence": cat["confidence"]},
                "result": p2(frame=frame, category=vec)}
    from models.recommendation.predict import predict as p3
    return {"model": name, "upstream": {"label": cat["label"],
                                        "confidence": cat["confidence"]},
            "result": p3(frame=frame, category=cat["label"])}


def _lan_address():
    """
    This machine's address on the local network.

    Opening a UDP socket toward a routable address makes the OS pick the
    interface it would actually use, without sending anything. Reading
    hostname resolution instead tends to return 127.0.1.1 on Debian systems.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


def cascade(subject, description):
    """
    Run Arm B's three models in their dependency order, keeping the hand-offs
    visible: B1 feeds its category distribution to B2, and its
    predicted label to B3 as a retrieval filter.
    """
    from shared.serving import ticket_frame
    from models.classification.predict import predict as p1
    from models.routing.predict import predict as p2
    from models.recommendation.predict import predict as p3

    if not (subject or "").strip() and not (description or "").strip():
        return {"error": "Enter a subject or a description."}

    frame = ticket_frame(subject, description)
    category = p1(frame=frame)
    routing = p2(frame=frame, category=category.pop("proba_vector"))
    retrieval = p3(frame=frame, category=category["label"])
    return {"category": category, "routing": routing, "retrieval": retrieval}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--lan", action="store_true",
                    help="serve on the local network, not just this machine")
    ap.add_argument("--auth", nargs="?", const="generate", default=None,
                    metavar="USER:PASS",
                    help="require a password off-loopback; bare --auth generates one")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args()
    if args.lan:
        args.host = "0.0.0.0"
    if args.auth:
        if args.auth == "generate":
            AUTH["user"], AUTH["password"] = "demo", secrets.token_urlsafe(9)
        elif ":" in args.auth:
            user, password = args.auth.split(":", 1)
            # An empty half would leave AUTH["password"] falsy, and a falsy
            # password means no checking at all -- the server would stand open
            # while whoever typed this believed it was shut. Refuse instead.
            if not user or not password:
                ap.error("--auth needs both halves, as user:password")
            AUTH["user"], AUTH["password"] = user, password
        else:
            ap.error("--auth takes user:password, or nothing at all")

    print("loading the deployed bundle ...")
    b = load()
    print(f"  trained on {b['meta']['n_train']} tickets, exported {b['meta']['exported']}")
    for task, spec in b["tasks"].items():
        print(f"  {task}: {len(spec['branches'])} branches, {spec['rule']} vote, "
              f"{len(spec['classes'])} classes")
    print(f"  retrieval index: {b['retrieval']['matrix'].shape[0]} resolved tickets")

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    lan_url = f"http://{_lan_address()}:{args.port}/" if args.host == "0.0.0.0" else None
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  running at {url}\n")
    print(f"    Arm A   end-to-end          {url}")
    print(f"    Arm B   B1 category         {url}classification")
    print(f"    Arm B   B2 routing          {url}routing")
    print(f"    Arm B   B3 resolutions      {url}recommendation")
    print(f"    Arm B   cascade             {url}cascade")
    print(f"            findings            {url}dashboard"
          + ("" if os.path.exists(DASHBOARD) else "   [not built - make dashboard]"))
    if AUTH["password"]:
        print(f"\n  password required off this machine", flush=True)
        print(f"    user      {AUTH['user']}")
        print(f"    password  {AUTH['password']}")
        print(f"  (shown once; loopback is exempt)", flush=True)
    if lan_url:
        print(f"\n  on this machine   {url}")
        print(f"  on the network    {lan_url}")
        if not AUTH["password"]:
            print("\n  note: no authentication -- anyone who can reach this port can"
                  " use it.\n  add --auth before exposing this beyond the local network.")
    print("\n  press Ctrl+C to stop\n")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
