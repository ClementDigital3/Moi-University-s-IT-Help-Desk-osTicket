"""
Synthetic osTicket-style corpus for the Moi University ICT help desk study.

Produces two files:
  data/raw/osticket_export.csv          -- fields exactly as per Appendix II of the
                                           thesis; this is what an authorised export
                                           would plausibly look like.
  data/raw/ground_truth_problem_ids.csv -- ticket_id -> latent_problem_id, used ONLY
                                           as the relevance key for Objective 4.

The latent problem id is deliberately kept out of the export: in a real deployment
the relevance ground truth for resolution recommendation would come from the human
relevance assessment allowed for in Sec. 3.14.

IMPORTANT: this corpus is a stand-in pending institutional authorisation (Sec. 1.10,
Sec. 3.4). Every figure derived from it must be reported as such.
"""
import csv, random, sys, os, re
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from problems import P, CATEGORIES, TEAMS, GENERIC_SUBJECTS

SEED = 20260915
N_TICKETS = 4800
START = datetime(2024, 1, 8)
END = datetime(2025, 12, 19)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

rng = random.Random(SEED)

DEPARTMENTS = [
    ("School of Information Sciences", 9), ("School of Education", 11),
    ("School of Business and Economics", 10), ("School of Engineering", 8),
    ("School of Medicine", 7), ("School of Arts and Social Sciences", 9),
    ("School of Law", 4), ("School of Science", 7),
    ("Registrar (Academic)", 8), ("Finance Department", 7),
    ("Human Resources", 4), ("University Library", 5),
    ("Procurement", 3), ("Estates and Works", 3), ("ICT Directorate", 5),
]

# Staff codes per team -- synthetic identifiers, not real personnel.
STAFF = {
    "Network Team":                ["ICT-NET-01", "ICT-NET-02", "ICT-NET-03"],
    "Systems & Accounts Team":     ["ICT-SYS-01", "ICT-SYS-02", "ICT-SYS-03", "ICT-SYS-04"],
    "eLearning Support Unit":      ["ICT-ELN-01", "ICT-ELN-02"],
    "MIS / SIS Support Team":      ["ICT-MIS-01", "ICT-MIS-02", "ICT-MIS-03"],
    "Applications Support Team":   ["ICT-APP-01", "ICT-APP-02"],
    "Hardware & Maintenance Team": ["ICT-HWM-01", "ICT-HWM-02", "ICT-HWM-03"],
    "Help Desk (Tier 1)":          ["ICT-HD-01", "ICT-HD-02", "ICT-HD-03"],
}


# ---------------------------------------------------------------------------
# Compositional slots (v2).
#
# P_CONTEXT is the single most important parameter in this generator. It is the
# probability that a ticket contains the detail identifying which system is at
# fault. When context is omitted the ticket is genuinely ambiguous -- a human
# triager could not resolve it either -- which puts a real ceiling on achievable
# accuracy and stops the classification task being trivially separable.
# ---------------------------------------------------------------------------
P_CONTEXT = 0.62
P_GENERIC_SUBJECT = 0.18
LABEL_NOISE = 0.05

TIMEPHRASE = [
    "since this morning", "since yesterday", "for the past two days",
    "since last week", "since the semester started", "for a while now",
    "since the power went off", "from today morning", "since friday",
]
LOCATION = [
    "in my office", "in our department", "at the main campus",
    "in the staff room", "in the computer lab", "in lecture hall three",
    "at the library", "in our block", "in the administration block",
]
REQUEST = [
    "kindly assist", "please help", "kindly assist urgently",
    "your assistance will be appreciated", "kindly look into it",
    "i would appreciate your help", "please attend to this as soon as possible",
    "kindly treat as urgent",
]
EXTRA = [
    "i have already restarted the machine", "i have tried several times",
    "my colleague is not having the same problem",
    "this is affecting my work", "i have a class shortly",
    "the deadline is today", "i have tried on both my phone and laptop",
    "it was working fine last week",
]

# Mislabelling happens between plausibly confusable categories, not at random.
CONFUSABLE = {
    "Account & Access Management": ["Learning Management System",
                                    "Student Information System",
                                    "Email & Collaboration"],
    "Learning Management System": ["Account & Access Management",
                                   "Student Information System"],
    "Student Information System": ["Account & Access Management",
                                   "Learning Management System"],
    "Email & Collaboration": ["Account & Access Management", "Application Support"],
    "Application Support": ["Hardware & Equipment", "Email & Collaboration"],
    "Hardware & Equipment": ["Application Support", "Network & Connectivity"],
    "Network & Connectivity": ["Hardware & Equipment", "Student Information System"],
}


def compose(p, r):
    """Assemble one ticket from slots. Returns (subject, body, had_context)."""
    sents = []
    core = r.choice(p["cores"])
    had_ctx = r.random() < P_CONTEXT
    if had_ctx:
        ctx = r.choice(p["contexts"])
        if r.random() < 0.5:
            sents.append(f"{core} and {ctx}")
        else:
            sents.append(core)
            sents.append(ctx)
    else:
        sents.append(core)

    if r.random() < 0.45:
        sents[0] = f"{sents[0]} {r.choice(LOCATION)}"
    if r.random() < 0.50:
        sents[-1] = f"{sents[-1]} {r.choice(TIMEPHRASE)}"
    if r.random() < 0.35:
        sents.append(r.choice(EXTRA))
    if r.random() < 0.55:
        sents.append(r.choice(REQUEST))

    body = ". ".join(x.strip() for x in sents if x.strip())
    body = body[0].upper() + body[1:] + "."
    subject = (r.choice(GENERIC_SUBJECTS) if r.random() < P_GENERIC_SUBJECT
               else r.choice(p["subjects"]))
    return subject, body, had_ctx


# ---------------------------------------------------------------------------
# Routing distributions.
#
# This is the mechanism that makes resolver routing a DISTINCT task from
# category classification (Sec. 2.6). Where a pid appears here, the resolver is
# genuinely ambiguous given the category alone -- the model must use the text.
# The classic case is "cannot print": a software-side problem (APP-PRINT-SW)
# and a hardware-side problem (HW-PRINTER) share almost all their vocabulary
# but are owned by different teams.
# ---------------------------------------------------------------------------
ROUTING = {
    "ACC-PWD-RESET":   [("Systems & Accounts Team", .60), ("Help Desk (Tier 1)", .40)],
    "ACC-LOCKED":      [("Systems & Accounts Team", .74), ("Help Desk (Tier 1)", .26)],
    "LMS-LOGIN":       [("eLearning Support Unit", .63), ("Systems & Accounts Team", .37)],
    "APP-PRINT-SW":    [("Applications Support Team", .52), ("Hardware & Maintenance Team", .48)],
    "HW-PRINTER":      [("Hardware & Maintenance Team", .79), ("Applications Support Team", .21)],
    "SIS-FEE":         [("MIS / SIS Support Team", .71), ("Applications Support Team", .29)],
    "SIS-PORTAL-SLOW": [("MIS / SIS Support Team", .66), ("Network Team", .34)],
    "NET-SLOW":        [("Network Team", .81), ("Hardware & Maintenance Team", .19)],
    "HW-PC-SLOW":      [("Hardware & Maintenance Team", .76), ("Applications Support Team", .24)],
    "EML-MEETING":     [("Applications Support Team", .61), ("eLearning Support Unit", .39)],
    "APP-INSTALL":     [("Applications Support Team", .80), ("Help Desk (Tier 1)", .20)],
}
DEFAULT_TIER1 = 0.11   # routine spillover to Tier 1 for every other problem

ABBREV = [
    (r"\bplease\b", "pls"), (r"\bcannot\b", "cant"), (r"\bpassword\b", "pwd"),
    (r"\buniversity\b", "uni"), (r"\bdepartment\b", "dept"), (r"\burgently\b", "urgntly"),
    (r"\byou\b", "u"), (r"\bthank you\b", "thanx"), (r"\bcomputer\b", "comp"),
    (r"\binformation\b", "info"), (r"\bplatform\b", "platfom"),
]
GREETINGS = ["Dear ICT,", "Hello,", "Good morning,", "Hi,", "Dear Sir/Madam,",
             "Greetings,", "Good afternoon,"]
SIGNOFFS = ["Kindly assist.", "Regards.", "Thanks in advance.", "Kindly assist urgently.",
            "Your assistance will be appreciated.", "Awaiting your response.", "Thank you."]

# Registration peaks -- Sept/Oct and Jan/Feb -- so SIS load is seasonal.
PEAK_MONTHS = {1: 1.9, 2: 1.5, 9: 2.1, 10: 1.6}
SEASONAL_PIDS = {"SIS-REG", "SIS-FEE", "SIS-PORTAL-SLOW", "LMS-COURSE-MISSING", "ACC-PWD-RESET"}

RESOLUTION_HOURS = {
    "Account & Access Management": (0.3, 6),   "Network & Connectivity": (2, 48),
    "Email & Collaboration": (1, 24),          "Learning Management System": (1, 30),
    "Student Information System": (2, 72),     "Application Support": (3, 96),
    "Hardware & Equipment": (4, 168),
}


def typo(text, r):
    """Introduce a single realistic keyboard slip."""
    words = text.split()
    idx = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not idx:
        return text
    i = r.choice(idx)
    w = words[i]
    j = r.randrange(1, len(w) - 1)
    mode = r.random()
    if mode < 0.4:      # transpose
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    elif mode < 0.7:    # drop
        w = w[:j] + w[j + 1:]
    else:               # double
        w = w[:j] + w[j] + w[j:]
    words[i] = w
    return " ".join(words)


def noisify(subject, body, r):
    if r.random() < 0.42:
        body = f"{r.choice(GREETINGS)} {body}"   # salutation noise, class-irrelevant
    if r.random() < 0.55:
        body = f"{body} {r.choice(SIGNOFFS)}"
    for pat, rep in ABBREV:
        if r.random() < 0.13:
            body = re.sub(pat, rep, body, flags=re.I)
    if r.random() < 0.38:
        body = typo(body, r)
    if r.random() < 0.12:
        subject = typo(subject, r)
    if r.random() < 0.07:
        subject = subject.upper()
    elif r.random() < 0.10:
        subject = subject.lower()
    if r.random() < 0.15:
        body = body.replace(". ", ".  ")
    return subject.strip(), body.strip()


def weighted(pairs, r):
    tot = sum(w for _, w in pairs)
    x = r.uniform(0, tot)
    acc = 0
    for item, w in pairs:
        acc += w
        if x <= acc:
            return item
    return pairs[-1][0]


def pick_date(pid, r):
    for _ in range(40):
        d = START + timedelta(days=r.randrange((END - START).days),
                              hours=r.randrange(7, 19), minutes=r.randrange(60))
        boost = PEAK_MONTHS.get(d.month, 1.0) if pid in SEASONAL_PIDS else 1.0
        if r.random() < boost / 2.1:
            return d
    return d


def assign_team(pid, r):
    dist = ROUTING.get(pid)
    if dist:
        return weighted([(t, w) for t, w in dist], r)
    primary = next(p["team"] for p in P if p["pid"] == pid)
    return "Help Desk (Tier 1)" if r.random() < DEFAULT_TIER1 else primary


def build():
    rows, truth = [], []
    pid_pairs = [(p["pid"], p["weight"]) for p in P]
    by_pid = {p["pid"]: p for p in P}
    tid = 100001

    for _ in range(N_TICKETS):
        pid = weighted(pid_pairs, rng)
        p = by_pid[pid]
        subject, body, had_ctx = compose(p, rng)
        subject, body = noisify(subject, body, rng)

        # The category as FILED by the help desk, which is not always correct --
        # the label-reliability concern raised in Sec. 1.9.
        filed_category = p["category"]
        if rng.random() < LABEL_NOISE:
            filed_category = rng.choice(CONFUSABLE[p["category"]])

        dept = weighted(DEPARTMENTS, rng)
        team = assign_team(pid, rng)
        # A wrong first assignment that was later corrected -- the reassignment
        # overhead described in Sec. 1.2. The FINAL team is the routing label.
        reassigned = rng.random() < 0.13
        staff = rng.choice(STAFF[team])

        created = pick_date(pid, rng)
        lo, hi = RESOLUTION_HOURS[p["category"]]
        dur = rng.uniform(lo, hi) * (1.8 if reassigned else 1.0)
        resolved = created + timedelta(hours=dur)

        s = rng.random()
        if s < 0.885:
            status = "Closed"
        elif s < 0.935:
            status = "Resolved"
        elif s < 0.975:
            status = "In Progress"
        else:
            status = "Open"

        if status in ("Closed", "Resolved"):
            note = p["resolution"]
            if rng.random() < 0.25:
                note = note.rstrip(".") + "." + rng.choice(
                    [" User confirmed the issue was resolved.", " Ticket closed.",
                     " Advised the user to report any recurrence.", ""])
            res_date = resolved.strftime("%Y-%m-%d %H:%M")
        else:
            note, res_date = "", ""

        rows.append({
            "Ticket ID": f"MU-{tid}",
            "Subject": subject,
            "Description": body,
            "Category": filed_category,
            "Subcategory": p["subcategory"],
            "Department": dept,
            "Assigned Resolver": staff,
            "Assigned Team": team,
            "Status": status,
            "Resolution Notes": note,
            "Created": created.strftime("%Y-%m-%d %H:%M"),
            "Resolved": res_date,
            "Reassigned": "Yes" if reassigned else "No",

        })
        truth.append({"Ticket ID": f"MU-{tid}", "latent_problem_id": pid,
                      "had_context": "Yes" if had_ctx else "No",
                      "true_category": p["category"]})
        tid += 1

    inject_quality_issues(rows, truth)
    return rows, truth


def inject_quality_issues(rows, truth):
    """Realistic dirt, so that the Sec. 3.11 screening stage has real work to do."""
    n = len(rows)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = 0

    def take(frac):
        nonlocal cut
        k = int(n * frac)
        sel = idx[cut:cut + k]
        cut += k
        return sel

    for i in take(0.015):                       # empty / useless descriptions
        rows[i]["Description"] = rng.choice(["", " ", "as above", "see subject", "-"])
    for i in take(0.011):                       # missing category label
        rows[i]["Category"] = ""
    for i in take(0.016):                       # unassigned
        rows[i]["Assigned Resolver"] = ""
        rows[i]["Assigned Team"] = ""
    for i in take(0.030):                       # closed without a resolution note
        if rows[i]["Status"] in ("Closed", "Resolved"):
            rows[i]["Resolution Notes"] = ""
    for i in take(0.010):                       # test tickets
        rows[i]["Subject"] = rng.choice(["test", "Testing", "test ticket please ignore", "xxx"])
        rows[i]["Description"] = rng.choice(["test", "testing the system", "ignore this", ""])
        rows[i]["Category"] = rng.choice(["", rows[i]["Category"]])

    # exact duplicate submissions (user pressed submit twice)
    dupes, dupe_truth = [], []
    tmap = {t["Ticket ID"]: (t["latent_problem_id"], t["had_context"],
                             t["true_category"]) for t in truth}
    maxid = max(int(r["Ticket ID"].split("-")[1]) for r in rows)
    for k, i in enumerate(take(0.020)):
        d = dict(rows[i])
        new_id = f"MU-{maxid + k + 1}"
        d["Ticket ID"] = new_id
        dupes.append(d)
        src = tmap[rows[i]["Ticket ID"]]
        dupe_truth.append({"Ticket ID": new_id, "latent_problem_id": src[0],
                           "had_context": src[1], "true_category": src[2]})
    rows.extend(dupes)
    truth.extend(dupe_truth)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, truth = build()
    rng.shuffle(rows)
    order = {r["Ticket ID"]: i for i, r in enumerate(rows)}
    truth.sort(key=lambda t: order[t["Ticket ID"]])

    exp = os.path.join(OUT, "osticket_export.csv")
    with open(exp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gt = os.path.join(OUT, "ground_truth_problem_ids.csv")
    with open(gt, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Ticket ID", "latent_problem_id",
                                          "had_context", "true_category"])
        w.writeheader()
        w.writerows(truth)

    print(f"tickets written : {len(rows)}")
    print(f"export          : {exp}")
    print(f"ground truth    : {gt}")


if __name__ == "__main__":
    main()
