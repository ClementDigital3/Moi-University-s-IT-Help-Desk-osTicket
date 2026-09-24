"""Fetch the sentence-transformers wheel set using pip's resolved report."""
import json, os, subprocess, sys, urllib.parse

REPORT = "/tmp/claude-1000/-home-creativetech/60a9e31c-79f8-4091-a87b-1caed03a8bdc/scratchpad/st_report.json"
WHEELS = "/home/creativetech/msc-ticket-ml/wheels"
HAVE = {"numpy", "scipy", "scikit-learn", "pillow", "packaging", "joblib"}


def size_of(url):
    out = subprocess.run(["curl", "-sIL", "--connect-timeout", "15", url],
                         capture_output=True, text=True).stdout
    for line in reversed(out.splitlines()):
        if line.lower().startswith("content-length:"):
            return int(line.split(":")[1].strip())
    return 0


def main():
    d = json.load(open(REPORT))
    os.makedirs(WHEELS, exist_ok=True)
    jobs = []
    for it in d["install"]:
        name = it["metadata"]["name"].lower()
        if name in HAVE:
            continue
        url = it["download_info"]["url"]
        fn = urllib.parse.unquote(url.split("/")[-1].split("#")[0])
        jobs.append((name, url, os.path.join(WHEELS, fn)))

    print(f"{len(jobs)} wheels to fetch")
    total = 0
    sized = []
    for name, url, dest in jobs:
        s = size_of(url)
        sized.append((name, url, dest, s))
        total += s
    print(f"total: {total/1e6:.1f} MB\n")

    failed = []
    for name, url, dest, s in sorted(sized, key=lambda t: t[3]):
        if os.path.exists(dest) and s and os.path.getsize(dest) == s:
            print(f"  have  {os.path.basename(dest)[:60]}")
            continue
        ok = False
        for attempt in range(1, 7):
            subprocess.run(["curl", "-L", "-C", "-", "--retry", "15",
                            "--retry-all-errors", "--retry-delay", "2",
                            "--connect-timeout", "15", "--speed-time", "30",
                            "--speed-limit", "1024", "-o", dest, url],
                           capture_output=True)
            got = os.path.getsize(dest) if os.path.exists(dest) else 0
            if got and (not s or got == s):
                print(f"  ok    {os.path.basename(dest)[:56]} ({got/1e6:.1f} MB)")
                ok = True
                break
            print(f"  retry {os.path.basename(dest)[:50]} ({got/1e6:.1f}/{s/1e6:.1f} MB) #{attempt}")
        if not ok:
            failed.append(os.path.basename(dest))
    print(f"\ncomplete: {len(sized)-len(failed)}/{len(sized)}")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
