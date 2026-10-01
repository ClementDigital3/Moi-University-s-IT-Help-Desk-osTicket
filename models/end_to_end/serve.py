"""
Local web prototype for the end-to-end model.

A single self-contained page served from the Python standard library. No web
framework, no CDN, no network access -- it runs on a laptop with the wifi off,
which is the only assumption worth making about a presentation room.

    .venv/bin/python -m models.end_to_end.serve
    .venv/bin/python -m models.end_to_end.serve --port 8080 --open

The bundle is loaded once at startup, so the first prediction is as fast as the
hundredth. Export it first with models.end_to_end.export.
"""
import os, json, argparse, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from shared.paths import ROOT
from .predict import load, predict

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "app.html")
CASCADE_PAGE = os.path.join(ROOT, "models", "cascade.html")
MODEL_PAGE = os.path.join(ROOT, "models", "model.html")

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


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
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
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = urlparse(self.path).path
        model_api = path.rsplit("/", 1)[-1] if path.startswith("/api/model/") else None
        if path not in ("/api/predict", "/api/cascade") and \
                model_api not in MODEL_ROUTES.values():
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            subj, desc = payload.get("subject", ""), payload.get("description", "")
            if path == "/api/predict":
                return self._send(200, json.dumps(predict(subj, desc)))
            if model_api:
                return self._send(200, json.dumps(run_model(model_api, subj, desc)))
            return self._send(200, json.dumps(cascade(subj, desc)))
        except SystemExit as e:                     # a bundle has not been exported
            self._send(200, json.dumps({"error": str(e)}))
        except Exception as e:                      # keep the demo alive
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


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

    def log_message(self, fmt, *args):
        if "/api/predict" in (args[0] if args else ""):
            print(f"  scored a ticket  [{self.log_date_time_string()}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--open", action="store_true", help="open a browser window")
    args = ap.parse_args()

    print("loading the deployed bundle ...")
    b = load()
    print(f"  trained on {b['meta']['n_train']} tickets, exported {b['meta']['exported']}")
    for task, spec in b["tasks"].items():
        print(f"  {task}: {len(spec['branches'])} branches, {spec['rule']} vote, "
              f"{len(spec['classes'])} classes")
    print(f"  retrieval index: {b['retrieval']['matrix'].shape[0]} resolved tickets")

    url = f"http://{args.host}:{args.port}/"
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"\n  running at {url}\n")
    print(f"    Arm A   end-to-end          {url}")
    print(f"    Arm B   B1 category         {url}classification")
    print(f"    Arm B   B2 routing          {url}routing")
    print(f"    Arm B   B3 resolutions      {url}recommendation")
    print(f"    Arm B   cascade             {url}cascade")
    print(f"            findings            {url}dashboard"
          + ("" if os.path.exists(DASHBOARD) else "   [not built - make dashboard]"))
    print("\n  press Ctrl+C to stop\n")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
