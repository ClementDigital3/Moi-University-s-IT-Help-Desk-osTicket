"""
Build the Chapter Four evidence dashboard as ONE self-contained HTML file.

No server, no CDN, no network. The data is embedded, so the file can be opened
by double-clicking it, carried on a USB stick, or emailed to a supervisor and it
will still work.

Run:    .venv/bin/python -m analysis.build_dashboard
Output: results/dashboard.html
"""
import os, json, time

from shared.paths import RESULTS, ROOT, ensure_dirs
from analysis.dashboard_data import collect

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "dashboard_template.html")
OUT = os.path.join(RESULTS, "dashboard.html")
TOKEN = "/*__DATA__*/{}"


def main():
    ensure_dirs()
    payload = collect()

    missing = [k for k, v in payload.items() if not v]
    if missing:
        print(f"  [warn] empty sections (run the pipeline first?): {missing}")

    html = open(TEMPLATE, encoding="utf-8").read()
    if TOKEN not in html:
        raise SystemExit(f"data placeholder not found in {TEMPLATE}")

    # </script> inside embedded JSON would close the tag early
    blob = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = html.replace(TOKEN, blob, 1)
    html = html.replace("</footer>",
                        f"Built {time.strftime('%Y-%m-%d %H:%M')}.</footer>", 1)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)

    kb = os.path.getsize(OUT) / 1024
    print(f"dashboard -> {os.path.relpath(OUT, ROOT)}  ({kb:.0f} KB, self-contained)")
    print(f"  sections: {', '.join(k for k, v in payload.items() if v)}")
    print(f"\nopen it with:  xdg-open {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
