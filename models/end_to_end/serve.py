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
        if path == "/api/examples":
            return self._send(200, json.dumps(EXAMPLES))
        self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/predict":
            return self._send(404, json.dumps({"error": "not found"}))
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            result = predict(payload.get("subject", ""),
                             payload.get("description", ""))
            self._send(200, json.dumps(result))
        except Exception as e:                      # keep the demo alive
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

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
    print(f"\n  ICT Help Desk triage prototype running at  {url}")
    print(f"  evidence dashboard                         {url}dashboard"
          + ("" if os.path.exists(DASHBOARD) else "   [not built yet]"))
    print("  press Ctrl+C to stop\n")
    if args.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
