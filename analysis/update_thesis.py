"""
Apply the agreed thesis revisions:
  1. Objective 3 (Sec. 1.4.2)  -- state the specialist-vs-unified comparison
  2. Research Question 3 (Sec. 1.5) -- match Objective 3
  3. Table 3.2 (Sec. 3.12)     -- add the integrated/unified candidate row

The source document is opened read-only; a revised COPY is written. Nothing in
Downloads is modified.
"""
import zipfile, shutil, re, os, random, sys

SRC = "/home/creativetech/Downloads/MSc Thesis - ICT Help Desk Ticket Management.docx"
DST = "/home/creativetech/msc-ticket-ml/docs/MSc Thesis - rev1 unified-model comparison.docx"

OBJ3_OLD = ("To develop and comparatively evaluate machine-learning approaches "
            "for ticket classification and resolver routing.")
OBJ3_NEW = ("To develop and comparatively evaluate machine-learning approaches for "
            "ticket classification and resolver routing, comparing task-specific "
            "models trained independently for each task against a single integrated "
            "model that performs both tasks from a shared representation.")

RQ3_OLD = ("How do alternative machine-learning approaches compare in the "
           "classification and resolver routing of ICT help desk tickets?")
RQ3_NEW = ("How do alternative machine-learning approaches compare in the "
           "classification and resolver routing of ICT help desk tickets, and how "
           "does a single integrated model compare with task-specific models "
           "trained independently for each task?")

ROW_LABEL = "Integrated / Unified"
ROW_BODY = ("Single shared-representation model performing ticket classification, "
            "resolver routing and historical-resolution recommendation from one "
            "encoding, evaluated against the task-specific models above")


def para_id():
    return "".join(random.choice("0123456789ABCDEF") for _ in range(8))


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_row(template: str) -> str:
    """
    Clone the template row and rewrite its two cells.

    Cells may hold several runs (Word splits runs around spell-check markers),
    so the text of the FIRST run in each cell is replaced and any remaining
    runs in that cell are emptied, rather than matching runs positionally.
    """
    row = re.sub(r"<w:proofErr[^>]*/>", "", template)
    row = re.sub(r'w14:paraId="[0-9A-Fa-f]{8}"',
                 lambda m: f'w14:paraId="{para_id()}"', row)

    cells = list(re.finditer(r"<w:tc>.*?</w:tc>", row, re.S))
    if len(cells) != 2:
        raise RuntimeError(f"expected 2 cells in template row, found {len(cells)}")

    values = [ROW_LABEL, ROW_BODY]
    parts, last = [], 0
    for ci, m in enumerate(cells):
        parts.append(row[last:m.start()])
        seen = [0]

        def sub(mm, ci=ci, seen=seen):
            seen[0] += 1
            return mm.group(1) + (esc(values[ci]) if seen[0] == 1 else "") + mm.group(3)

        parts.append(re.sub(r"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)", sub, m.group(0), flags=re.S))
        last = m.end()
    parts.append(row[last:])
    return "".join(parts)


def main():
    if not os.path.exists(SRC):
        sys.exit(f"source not found: {SRC}")
    os.makedirs(os.path.dirname(DST), exist_ok=True)

    zin = zipfile.ZipFile(SRC)
    doc = zin.read("word/document.xml").decode("utf-8")
    original = doc
    changes = []

    # ---- 1 & 2: objective and research question -------------------------
    for label, old, new in (("Objective 3 (Sec. 1.4.2)", OBJ3_OLD, OBJ3_NEW),
                            ("Research Question 3 (Sec. 1.5)", RQ3_OLD, RQ3_NEW)):
        n = doc.count(old)
        if n != 1:
            print(f"  [WARN] {label}: expected 1 occurrence, found {n} -- skipped")
            continue
        doc = doc.replace(old, new)
        changes.append(label)

    # ---- 3: Table 3.2 new row ------------------------------------------
    anchor = doc.find("Contextual / Semantic Benchmark")
    if anchor == -1:
        print("  [WARN] Table 3.2 anchor not found -- row not added")
    else:
        start = doc.rfind("<w:tr ", 0, anchor)
        if start == -1:
            start = doc.rfind("<w:tr>", 0, anchor)
        end = doc.find("</w:tr>", anchor) + len("</w:tr>")
        template = doc[start:end]
        new_row = build_row(template)
        doc = doc[:end] + new_row + doc[end:]
        changes.append("Table 3.2 (Sec. 3.12) -- integrated/unified row added")

    if doc == original:
        sys.exit("no changes applied; aborting without writing")

    # ---- repackage -------------------------------------------------------
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = doc.encode("utf-8") if item.filename == "word/document.xml" \
                   else zin.read(item.filename)
            zout.writestr(item, data)
    zin.close()

    print("Applied:")
    for c in changes:
        print(f"  - {c}")
    print(f"\nsource (unchanged): {SRC}")
    print(f"revised copy      : {DST}")


if __name__ == "__main__":
    random.seed(7)
    main()
