"""
The dashboard's navigation must match the component's.

Every other page renders its menu from window.Nav, so adding a view is one
edit. The dashboard cannot: it is opened straight from results/ as often as it
is served, and a nav that depended on /static/ui.js would be empty when it is.
So its menu is written out by hand, which means it can drift -- and silently,
because a missing link looks like a design choice rather than a bug.

This compares the two and fails if they disagree. It is the reason the
duplication is acceptable rather than merely convenient.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def component_links():
    """The hrefs in window.Nav's VIEWS list, in order."""
    js = (ROOT / "shared" / "ui.js").read_text(encoding="utf-8")
    block = re.search(r"const VIEWS = \[(.*?)\];", js, re.S)
    if not block:
        sys.exit("could not find VIEWS in shared/ui.js")
    return re.findall(r'href:\s*"([^"]+)"', block.group(1))


def dashboard_links():
    html = (ROOT / "analysis" / "dashboard_template.html").read_text(encoding="utf-8")
    nav = re.search(r'<nav class="nav".*?</nav>', html, re.S)
    if not nav:
        sys.exit("could not find the nav in analysis/dashboard_template.html")
    return re.findall(r'href="([^"]+)"', nav.group(0))


def main():
    want, got = component_links(), dashboard_links()
    if want == got:
        print(f"  nav consistent — {len(want)} views in the same order")
        return 0
    print("  the dashboard's nav does not match shared/ui.js\n")
    print(f"    component: {want}")
    print(f"    dashboard: {got}\n")
    for missing in [h for h in want if h not in got]:
        print(f"    missing from the dashboard: {missing}")
    for extra in [h for h in got if h not in want]:
        print(f"    only in the dashboard:      {extra}")
    if sorted(want) == sorted(got):
        print("    (same links, different order)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
