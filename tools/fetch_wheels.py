"""
Resumable wheel fetcher.

pip stalled indefinitely on files.pythonhosted.org without timing out. It had,
however, already resolved the full dependency graph, so we take the exact wheel
list from its log and fetch each file ourselves with curl -C - (resume) and
aggressive retries, then install offline from the local directory.
"""
import json, os, re, subprocess, sys, urllib.request

ROOT = "/home/creativetech/msc-ticket-ml"
WHEELS = os.path.join(ROOT, "wheels")
SKIP = {"nvidia_nccl_cu13"}          # GPU-only, not needed for CPU XGBoost


def wheels_from_log(path):
    names = sorted(set(re.findall(r"[A-Za-z0-9_.+-]+\.whl", open(path).read())))
    out = []
    for n in names:
        n = n.replace(".metadata", "")
        stem = n.split("-")[0]
        if stem in SKIP:
            print(f"  skip (GPU-only): {n}")
            continue
        ver = n.split("-")[1]
        out.append((stem.replace("_", "-"), ver, n))
    return out


def url_for(project, version, filename):
    api = f"https://pypi.org/pypi/{project}/{version}/json"
    with urllib.request.urlopen(api, timeout=30) as r:
        data = json.load(r)
    for u in data["urls"]:
        if u["filename"] == filename:
            return u["url"], u["size"]
    for u in data["urls"]:                     # fall back to any compatible wheel
        if u["packagetype"] == "bdist_wheel":
            return u["url"], u["size"]
    raise RuntimeError(f"no wheel url for {filename}")


def fetch(url, dest, size):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"  have  {os.path.basename(dest)}")
        return True
    cmd = ["curl", "-L", "-C", "-", "--retry", "15", "--retry-all-errors",
           "--retry-delay", "2", "--connect-timeout", "15",
           "--speed-time", "30", "--speed-limit", "1024",
           "-o", dest, url]
    for attempt in range(1, 6):
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"  ok    {os.path.basename(dest)}  ({size/1e6:.1f} MB)")
            return True
        got = os.path.getsize(dest) if os.path.exists(dest) else 0
        print(f"  retry {os.path.basename(dest)} ({got/1e6:.1f}/{size/1e6:.1f} MB, attempt {attempt})")
    return False


def main():
    os.makedirs(WHEELS, exist_ok=True)
    items = wheels_from_log(os.path.join(ROOT, "install.log"))
    print(f"{len(items)} wheels to fetch\n")
    total = 0
    targets = []
    for proj, ver, fn in items:
        try:
            u, s = url_for(proj, ver, fn)
            targets.append((u, os.path.join(WHEELS, fn), s))
            total += s
        except Exception as e:
            print(f"  [!] {proj} {ver}: {type(e).__name__}: {e}")
    print(f"total download: {total/1e6:.1f} MB\n")

    failed = []
    for u, d, s in sorted(targets, key=lambda t: t[2]):    # small files first
        if not fetch(u, d, s):
            failed.append(os.path.basename(d))
    print(f"\ncomplete: {len(targets)-len(failed)}/{len(targets)}")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
