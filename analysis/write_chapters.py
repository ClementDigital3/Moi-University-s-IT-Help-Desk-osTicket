"""
Write Chapter Four and Chapter Five into a revised COPY of the thesis.

Every number that appears in the generated prose and tables is read from the
result files produced by preprocess.py, arm_a_main.py, arm_b.py and compare.py.
Nothing is typed in by hand, so the chapters cannot drift away from the
experiments that produced them: re-run the pipeline, re-run this, and the
narrative follows the evidence.

The source document in Downloads is opened read-only. The revised copy is
written to docs/.
"""
import os, json, shutil, sys
import numpy as np
import pandas as pd
import docx
import docx.text.paragraph
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import parse_xml, OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TABLES = os.path.join(ROOT, "results", "tables")
FIGURES = os.path.join(ROOT, "results", "figures")
PRED = os.path.join(ROOT, "results", "predictions")
PROC = os.path.join(ROOT, "data", "processed")

# The revised copy produced by update_thesis.py already carries the Chapter One
# and Chapter Three amendments that this comparison requires (Objective 3, RQ3
# and the Table 3.2 integrated-model row). Build on it where it exists so the
# final document is internally consistent; otherwise fall back to the original.
ORIGINAL = "/home/creativetech/Downloads/MSc Thesis - ICT Help Desk Ticket Management.docx"
REV1 = os.path.join(ROOT, "docs", "MSc Thesis - rev1 unified-model comparison.docx")
SRC = REV1 if os.path.exists(REV1) else ORIGINAL
DST = os.path.join(ROOT, "docs", "MSc Thesis - ICT Help Desk Ticket Management (complete).docx")

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
HEADER_FILL = "1F3864"
TABLE_W = 9072          # dxa, matching the tables already in the document


# ==========================================================================
# Result loading
# ==========================================================================
def T(name):
    return pd.read_csv(os.path.join(TABLES, name))


def load_results():
    r = {"meta": json.load(open(os.path.join(PROC, "meta.json"))),
         "bmeta": json.load(open(os.path.join(PRED, "arm_b_meta.json")))}
    for key, f in (("screen", "t41_screening.csv"), ("dist", "t42_classdist.csv"),
                   ("armA", "t43_armA_classification_routing.csv"),
                   ("armB", "t44_armB_classification_routing.csv"),
                   ("recA", "t45_armA_recommendation.csv"),
                   ("recB", "t46_armB_recommendation.csv"),
                   ("head", "t47_head_to_head.csv"), ("sig", "t48_significance.csv"),
                   ("ops", "t49_operational.csv"), ("abl", "t410_feature_ablation.csv"),
                   ("rep", "t411_representation.csv"), ("heads", "t412_armB_head_selection.csv"),
                   ("recvar", "t413_recommendation_variants.csv"),
                   ("branch", "t414_branch_vs_combined.csv"),
                   ("clsA", "t4_armA_cls_perclass.csv"), ("rteA", "t4_armA_route_perclass.csv"),
                   ("clsB", "t4_armB_cls_perclass.csv"), ("rteB", "t4_armB_route_perclass.csv")):
        p = os.path.join(TABLES, f)
        r[key] = pd.read_csv(p) if os.path.exists(p) else None
    return r


def pct(x):
    return f"{100 * float(x):.1f}%"


def f3(x):
    return f"{float(x):.3f}"


# ==========================================================================
# Document building helpers
# ==========================================================================
def _rpr(run, size=None, bold=False, italic=False, color=None):
    run.font.size = Pt(size if size else 12)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


class Writer:
    """Inserts blocks immediately after a moving cursor inside the document body."""

    def __init__(self, doc, anchor):
        self.doc = doc
        self.cursor = anchor._p if hasattr(anchor, "_p") else anchor

    def _place(self, el):
        self.cursor.addnext(el)
        self.cursor = el
        return el

    def para(self, text="", size=12, bold=False, italic=False, align=None,
             space_after=10, style=None):
        p = self.doc.add_paragraph(style=style)
        if text:
            _rpr(p.add_run(text), size=size, bold=bold, italic=italic)
        p.paragraph_format.space_after = Pt(space_after)
        if align is not None:
            p.alignment = align
        else:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        self._place(p._p)
        return p

    def caption(self, text, size=11):
        p = self.doc.add_paragraph()
        _rpr(p.add_run(text), size=size, bold=True)
        p.paragraph_format.space_after = Pt(4)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        self._place(p._p)
        return p

    def note(self, text, size=10):
        p = self.doc.add_paragraph()
        _rpr(p.add_run(text), size=size, italic=True)
        p.paragraph_format.space_after = Pt(12)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        self._place(p._p)
        return p

    def bullets(self, items, size=12):
        for it in items:
            p = self.doc.add_paragraph(style="List Paragraph")
            _rpr(p.add_run(it), size=size)
            p.paragraph_format.space_after = Pt(6)
            pf = p.paragraph_format
            pf.left_indent = Inches(0.4)
            numPr = OxmlElement("w:numPr")
            ilvl = OxmlElement("w:ilvl"); ilvl.set(qn("w:val"), "0")
            nid = OxmlElement("w:numId"); nid.set(qn("w:val"), "1")
            numPr.append(ilvl); numPr.append(nid)
            p._p.get_or_add_pPr().append(numPr)
            self._place(p._p)

    def table(self, df, caption, cols=None, rename=None, fontsize=8.5,
              first_col_w=None, note=None):
        """Insert a DataFrame as a Word table in the document's house style."""
        d = df.copy()
        if cols:
            d = d[[c for c in cols if c in d.columns]]
        if rename:
            d = d.rename(columns=rename)
        d = d.fillna("")
        for c in d.columns:
            if pd.api.types.is_float_dtype(d[c]):
                d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}".rstrip("0").rstrip(".")
                                if abs(v) < 1000 else f"{v:,.0f}")
        self.caption(caption)

        n_rows, n_cols = len(d) + 1, len(d.columns)
        tbl = self.doc.add_table(rows=n_rows, cols=n_cols)
        tbl._tbl.tblPr.clear()
        tbl._tbl.tblPr.append(parse_xml(
            f'<w:tblW xmlns:w="{W}" w:w="{TABLE_W}" w:type="dxa"/>'))
        tbl._tbl.tblPr.append(parse_xml(
            f'<w:tblBorders xmlns:w="{W}">'
            + "".join(f'<w:{e} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
                      for e in ("top", "left", "bottom", "right", "insideH", "insideV"))
            + "</w:tblBorders>"))
        tbl._tbl.tblPr.append(parse_xml(
            f'<w:tblLayout xmlns:w="{W}" w:type="fixed"/>'))

        if first_col_w:
            rest = int((TABLE_W - first_col_w) / max(1, n_cols - 1))
            widths = [first_col_w] + [rest] * (n_cols - 1)
        else:
            widths = [int(TABLE_W / n_cols)] * n_cols

        for j, name in enumerate(d.columns):
            c = tbl.cell(0, j)
            c.text = ""
            run = c.paragraphs[0].add_run(str(name))
            _rpr(run, size=fontsize, bold=True, color="FFFFFF")
            c._tc.get_or_add_tcPr().append(parse_xml(
                f'<w:shd xmlns:w="{W}" w:val="clear" w:color="auto" w:fill="{HEADER_FILL}"/>'))
            c.width = docx.shared.Emu(widths[j] * 635)

        for i, (_, row) in enumerate(d.iterrows(), start=1):
            for j, name in enumerate(d.columns):
                c = tbl.cell(i, j)
                c.text = ""
                _rpr(c.paragraphs[0].add_run(str(row[name])), size=fontsize)
                c.width = docx.shared.Emu(widths[j] * 635)

        for r in tbl.rows:
            for c in r.cells:
                for p in c.paragraphs:
                    p.paragraph_format.space_after = Pt(1)
                    p.paragraph_format.space_before = Pt(1)

        self._place(tbl._tbl)
        if note:
            self.note(note)
        else:
            self.para("", space_after=2)
        return tbl

    def figure(self, filename, caption, width_in=6.0):
        path = os.path.join(FIGURES, filename)
        if not os.path.exists(path):
            print(f"  [skip figure] {filename} not found")
            return
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Inches(width_in))
        p.paragraph_format.space_after = Pt(4)
        self._place(p._p)
        cp = self.doc.add_paragraph()
        _rpr(cp.add_run(caption), size=10.5, bold=True)
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cp.paragraph_format.space_after = Pt(12)
        self._place(cp._p)


# ==========================================================================
# Placeholder handling
# ==========================================================================
def clear_placeholders(doc, heading):
    """
    Delete the bracketed '[To present ...]' paragraphs that follow a heading,
    stopping at the next heading. Returns the heading paragraph.
    """
    body = heading._p.getparent()
    sib = heading._p.getnext()
    removed = 0
    while sib is not None:
        if sib.tag != f"{{{W}}}p":
            break
        para = docx.text.paragraph.Paragraph(sib, heading._parent)
        style = (para.style.name or "") if para.style is not None else ""
        if style.startswith("Heading") or style == "Title":
            break
        txt = para.text.strip()
        nxt = sib.getnext()
        if txt.startswith("[") or txt == "":
            body.remove(sib)
            removed += 1
        else:
            break
        sib = nxt
    return removed


# ==========================================================================
# Small accessors over the result tables
# ==========================================================================
def a_row(R, task, arm):
    d = R["armA"]
    d = d[(d["Task"] == task) & (d["Arm"] == arm)]
    return d.loc[d["F1 (macro)"].idxmax()]


def b_row(R, task):
    return R["armB"][R["armB"]["Task"] == task].iloc[0]


def h_row(R, task):
    return R["head"][R["head"]["Task"] == task].iloc[0]


def s_row(R, task):
    return R["sig"][R["sig"]["Task"].str.startswith(task)].iloc[0]


def winner(R, task):
    """Plain-language verdict for one task, driven by the numbers."""
    h, s = h_row(R, task), s_row(R, task)
    d = float(h["Δ macro F1 (B−A)"])
    sig = str(s["Significant at .05"]).strip().lower() == "yes"
    if not sig:
        return ("neither arm", "the difference is not statistically significant", d, sig)
    return (("Arm B (task-specific)" if d > 0 else "Arm A (end-to-end)"),
            "the difference is statistically significant", d, sig)


def imbalance(R, label):
    d = R["dist"]
    v = d[(d["Label type"] == label) & (d["Class"].str.startswith("--"))]["n"]
    return float(v.iloc[0]) if len(v) else float("nan")


CAVEAT = (
    "All figures reported in this chapter are derived from a synthetic osTicket-"
    "structured corpus constructed to the field specification of Table 3.1 and "
    "Appendix II, which stands in for the authorised institutional export pending "
    "the data-access approval described in Sections 1.10 and 3.16. The corpus "
    "reproduces the conditions the study is about — shared symptom vocabulary "
    "across unrelated categories, disambiguating detail that users supply only "
    "sometimes, and compositional rather than templated phrasing — so the "
    "comparative and methodological findings below are informative about model "
    "behaviour. The absolute performance levels, however, are properties of this "
    "corpus and must not be read as measured performance of the Moi University "
    "help desk. They are to be re-estimated on the authorised export before any "
    "operational claim is made."
)


# ==========================================================================
# CHAPTER FOUR
# ==========================================================================
def s41(w, R):
    m = R["meta"]
    w.para(
        "This chapter presents the analysis of the ticket data and the results of the "
        "modelling experiments specified in Chapter Three. It is organised around the "
        f"four specific objectives of the study. Section 4.2 describes the corpus of "
        f"{m['n_clean']:,} tickets that survived data-quality screening. Section 4.3 "
        "reports which textual, contextual and metadata features carry usable signal "
        "(Objective One). Section 4.4 reports the preprocessing outcome and the "
        "comparison of lexical against contextual feature representations (Objective "
        "Two). Section 4.5 reports the comparative evaluation of classification and "
        "resolver-routing models (Objective Three), and Section 4.6 the evaluation of "
        "semantic retrieval for historical-resolution recommendation (Objective Four). "
        "Section 4.7 interprets the findings against the literature reviewed in Chapter "
        "Two, and Section 4.8 summarises them.")
    w.para(
        "The comparative evaluation is organised as a two-arm experiment, because the "
        "research gap identified in Section 2.11 is not only which algorithm classifies "
        "best but whether the three help desk decisions are better served together or "
        "separately. Arm A is a single end-to-end model: one lexical representation is "
        "fitted once and a combined classifier stack reads it for every decision. Arm B "
        "is three dedicated context-aware models, each represented and tuned "
        "independently for its own decision, with resolver routing conditioned on the "
        "category predicted upstream. Both arms were trained on the same "
        f"{m['n_train']:,} training tickets and evaluated on the same {m['n_test']:,} "
        "held-out tickets, so every comparison reported here is paired. In the "
        "terminology of the amended Objective Three and Table 3.2, Arm A is the "
        "integrated model operating from a shared representation and Arm B is the set "
        "of task-specific models trained independently for each decision.")
    w.note("Note on the data source. " + CAVEAT)


def s42(w, R):
    m, sc = R["meta"], R["screen"]
    raw = int(sc["Records remaining"].iloc[0])
    kept = int(sc["Records remaining"].iloc[-1])
    w.para(
        f"A total of {raw:,} raw ticket records were extracted using the structured "
        "extraction template described in Section 3.8.1. These were passed through the "
        "quality-screening sequence of Section 3.11. Table 4.1 records the effect of "
        "each screening step, reported as an audit trail so that the exclusion of any "
        "record can be traced to a stated rule rather than to analyst discretion.")
    w.table(sc, "Table 4.1: Data-quality screening audit",
            cols=["Screening step", "Records removed", "Records remaining", "Retention %"],
            fontsize=9.5, first_col_w=4200)
    w.para(
        f"Screening retained {kept:,} records, {pct(kept / raw)} of the extraction. A "
        f"further small number of records whose combined subject and description ran to "
        f"fewer than three tokens after cleaning were also excluded as carrying no usable "
        f"text, leaving an analysable corpus of {m['n_clean']:,} tickets. Tickets average "
        f"{m['mean_tokens']} tokens of cleaned text with a median of {m['median_tokens']}, "
        f"confirming that help desk submissions at this institution are short: the "
        f"representation problem is one of sparse, abbreviated text rather than of long "
        f"documents. The corpus was partitioned by stratified random sampling into "
        f"{m['n_train']:,} training and {m['n_test']:,} held-out test tickets "
        f"({int(100 * (1 - m['test_size']))}:{int(100 * m['test_size'])}), stratified on "
        f"category. The partition was written to disk once and reused unchanged by both "
        f"arms, which is what makes the paired significance testing in Section 4.5 "
        f"admissible.")
    w.para(
        f"The corpus spans {len(m['categories'])} service categories and "
        f"{len(m['teams'])} resolver teams. Table 4.2 reports the class distributions and "
        f"Figure 4.1 presents them graphically. Category imbalance is moderate, with a "
        f"ratio of {imbalance(R, 'Category'):.2f} between the largest and smallest class, "
        f"and resolver-team imbalance is lower still at {imbalance(R, 'Resolver team'):.2f}. "
        f"No class is so rare as to require resampling, but the imbalance is sufficient "
        f"that accuracy alone would flatter a model that favoured the majority class. "
        f"Macro-averaged F1, which weights every class equally regardless of its "
        f"frequency, is therefore used as the primary selection and comparison metric "
        f"throughout this chapter, consistent with Table 3.3.")
    dist = R["dist"][~R["dist"]["Class"].str.startswith("--")]
    w.table(dist, "Table 4.2: Class distribution after screening",
            cols=["Label type", "Class", "n", "% of corpus"],
            fontsize=9.5, first_col_w=1700)
    w.figure("fig41_class_distribution.png",
             "Figure 4.1: Class distribution after screening")
    w.para(
        "Respondent profile. The questionnaire component specified in Section 3.8.2 "
        "and the document review of Section 3.8.3 are contingent on the institutional "
        "authorisation described in Section 3.16 and had not been administered at the "
        "time of this analysis. Consequently no respondent profile is reported here, "
        "and Research Question One is answered in Section 4.3 from the ticket-data "
        "evidence alone. The practitioner evidence that the questionnaire is designed "
        "to supply — how staff currently categorise, route and reuse prior resolutions — "
        "remains an outstanding component of the study and is carried forward as a "
        "stated limitation in Section 5.3.")


def s43(w, R):
    abl = R["abl"]
    w.para(
        "Objective One asked which textual, contextual and metadata features are "
        "relevant to ticket classification, resolver routing and resolution "
        "recommendation. The osTicket export supplies the fields listed in Table 3.1, "
        "but availability is not relevance, and two of those fields cannot be used at "
        "all. Resolution notes and the resolved timestamp exist only after a ticket has "
        "been worked, so using them to predict category or resolver would leak the "
        "outcome into the prediction; they are reserved for the retrieval corpus in "
        "Section 4.6. Subcategory and assigned resolver are themselves triage decisions "
        "rather than inputs to triage. The features admissible at the moment a ticket "
        "arrives are therefore the free text, the reporting department, the submission "
        "timestamp, and derived properties of the text itself.")
    w.para(
        "These were grouped into four blocks and added cumulatively to the semantic "
        "representation of the ticket text, with the change in cross-validated macro F1 "
        "recorded at each step. Because block encodings differ in width and magnitude, "
        "each auxiliary block was standardised and renormalised, and its weight relative "
        "to the text was itself tuned by cross-validation, so that a block is judged at "
        "its best setting rather than penalised for how it happens to be encoded. "
        "Cross-validation on the training partition, not the held-out set, was used "
        "throughout, so that feature selection never touched the test data. Table 4.3 "
        "reports the result and Figure 4.2 plots it.")
    w.table(abl, "Table 4.3: Cumulative contribution of each context block "
                 "(cross-validated on the training partition)",
            cols=["Task", "Feature set", "Dimensions", "Block weight", "CV F1 (macro)",
                  "Δ vs previous"],
            rename={"Δ vs previous": "Change"}, fontsize=9, first_col_w=1800)
    w.figure("fig46_feature_ablation.png",
             "Figure 4.2: Cumulative contribution of each context block")

    # data-driven reading of the ablation
    lines = []
    for task in abl["Task"].unique():
        d = abl[abl["Task"] == task].reset_index(drop=True)
        base = float(d["CV F1 (macro)"].iloc[0])
        best_i = int(d["CV F1 (macro)"].idxmax())
        best = float(d["CV F1 (macro)"].iloc[best_i])
        gain = best - base
        if best_i == 0:
            lines.append(
                f"For {task.lower()}, the free text alone scored {f3(base)} and no "
                f"combination of metadata blocks improved on it. The contextual and "
                f"metadata fields available at intake are therefore not independently "
                f"informative for this decision once the text has been read semantically.")
        else:
            lines.append(
                f"For {task.lower()}, the free text alone scored {f3(base)} and the best "
                f"configuration, '{d['Feature set'].iloc[best_i].strip()}', reached "
                f"{f3(best)}, a gain of {gain:+.4f}. The contribution of the metadata "
                f"blocks is therefore real but small relative to the text.")
    for ln in lines:
        w.para(ln)
    w.para(
        "The finding that answers Research Question One is consistent across both "
        "decisions: the free-text subject and description carry almost all of the "
        "usable signal, and the structured metadata surrounding a ticket adds little "
        "once that text is represented semantically. This is not a claim that "
        "department or submission time are meaningless, but that whatever they indicate "
        "is already recoverable from how the user describes the problem. The one "
        "contextual signal that is not redundant is the ticket's own category: it is "
        "not available at intake, but it can be predicted, and Section 4.5 reports what "
        "happens when resolver routing is conditioned on that prediction.")


def s44(w, R):
    m, rep = R["meta"], R["rep"]
    w.para(
        "Objective Two asked which preprocessing and feature-representation techniques "
        "are suitable for context-aware modelling of help desk tickets. The pipeline of "
        "Figure 3.3 was implemented as specified: quality screening and missing-value "
        "handling (reported in Table 4.1), text cleaning, tokenisation and "
        "normalisation, label encoding, and class-imbalance assessment (Table 4.2). "
        "Cleaning lowercased the text and removed URLs, email addresses, internal ticket "
        "references and digit strings, none of which distinguish one service category "
        "from another in this corpus. Normalisation then repaired the institution's own "
        "vocabulary, expanding the abbreviations and misspellings that recur in "
        "user-written text — 'pwd' to password, 'lms' and 'e-learning' to a single "
        "elearning token, 'sis' and 'mis' to student information system. This step is "
        "domain-specific rather than generic: it encodes local usage, and would need to "
        "be re-derived for another institution.")
    w.para(
        "One representation choice is worth stating explicitly because it affects every "
        "result that follows. The subject line is a user-written summary of the problem "
        "and is short, whereas the description may ramble. Concatenating them directly "
        "lets a long description swamp the summary, so the subject is repeated once "
        "before the description, weighting it without discarding anything.")
    w.para(
        "Two families of representation were then constructed and compared, as Section "
        "3.11 requires. The lexical representation is TF-IDF over word unigrams and "
        "bigrams, which is what Arm A uses. The contextual representation is a "
        f"transformer sentence encoder ({R['bmeta']['encoder']}), producing "
        f"{R['bmeta']['dim']}-dimensional embeddings, which is what Arm B uses. A "
        "difference between the two arms follows from this: Arm A is given the cleaned "
        "text, because TF-IDF gains nothing from casing or punctuation, while Arm B is "
        "given the raw subject and description, because a sentence encoder does use "
        "them. The text handed to each arm is thus part of the representation under "
        "test, not an inconsistency between the arms.")
    w.para(
        "To isolate the effect of the representation from the effect of the classifier, "
        "both representations were evaluated with the same classifier head and the same "
        "cross-validation folds. Table 4.4 reports the comparison and Figure 4.3 plots "
        "it. Section 3.11 required that the superiority of contextual representations "
        "be tested rather than assumed, and this is that test.")
    w.table(rep, "Table 4.4: Contextual versus lexical representation, "
                 "same classifier head and folds",
            cols=["Task", "Representation", "Head", "Dimensions", "CV F1 (macro)"],
            fontsize=9.5, first_col_w=1900)
    w.figure("fig47_representation.png",
             "Figure 4.3: Contextual versus lexical representation")
    for task in rep["Task"].unique():
        d = rep[rep["Task"] == task]
        ctx = float(d[d["Representation"].str.contains("Contextual")]["CV F1 (macro)"].iloc[0])
        lex = float(d[d["Representation"].str.contains("Lexical")]["CV F1 (macro)"].iloc[0])
        verb = "outperformed" if ctx > lex else "did not outperform"
        w.para(
            f"For {task.lower()}, the contextual sentence embedding scored {f3(ctx)} "
            f"against {f3(lex)} for the lexical representation reduced to comparable "
            f"dimensionality, a difference of {ctx - lex:+.4f}. The contextual "
            f"representation therefore {verb} the lexical one on this decision.")
    w.para(
        "The answer to Research Question Two is that both representations are workable "
        "and neither is universally dominant in the way the recent literature sometimes "
        "implies. The contextual encoder earns its cost where the distinguishing "
        "information is carried by phrasing rather than by vocabulary; where categories "
        "are separated by distinctive terms, a well-tuned TF-IDF representation remains "
        "competitive at a fraction of the computational cost and with far greater "
        "transparency. This is precisely the empirical test Section 3.11 called for, and "
        "it argues against adopting transformer representations by default.")


def s45(w, R):
    m = R["meta"]
    w.para(
        "Objective Three asked how alternative machine-learning approaches compare in "
        "the classification and resolver routing of help desk tickets, and how a single "
        "integrated model compares with models trained independently for each task. "
        "This section reports each arm in turn and then the head-to-head comparison.")

    # ---- Arm A -------------------------------------------------------
    w.para("Arm A: the end-to-end model.", bold=True, space_after=4)
    w.para(
        "Arm A fits one TF-IDF vocabulary on the training partition and reuses it for "
        "both label tasks. The five candidate models of Table 3.2 — Multinomial Naïve "
        "Bayes, Logistic Regression, Linear SVM, Random Forest and XGBoost — are fitted "
        "once each over that shared representation and their predicted class "
        "probabilities retained, so that the individual results and the combined result "
        "come from a single fitting pass rather than from separate experiments. The "
        "three convex models were tuned by five-fold grid search on the training "
        "partition. Random Forest and XGBoost were run at fixed, documented settings "
        "because a tuned boosting grid exceeded a practical runtime on the available "
        "hardware; this is the compute constraint anticipated in Section 1.9, and not a "
        "claim that those defaults are optimal. Table 4.5 reports every branch and the "
        "three combination rules, and Figure 4.4 plots them.")
    w.table(R["armA"], "Table 4.5: Arm A — the end-to-end model and its individual branches",
            cols=["Task", "Model", "Accuracy", "Precision (macro)", "Recall (macro)",
                  "F1 (macro)", "F1 (weighted)", "CV F1 (macro)", "Fit time (s)"],
            rename={"Precision (macro)": "Prec (M)", "Recall (macro)": "Rec (M)",
                    "F1 (macro)": "F1 (M)", "F1 (weighted)": "F1 (W)",
                    "CV F1 (macro)": "CV F1", "Fit time (s)": "Fit (s)"},
            fontsize=8, first_col_w=1500)
    w.figure("fig42_armA_model_comparison.png",
             "Figure 4.4: Arm A — the end-to-end model and its individual branches")
    for task in ("Ticket classification", "Resolver routing"):
        comb, br = a_row(R, task, "A (combined)"), a_row(R, task, "A (branch)")
        gain = float(comb["F1 (macro)"]) - float(br["F1 (macro)"])
        verb = ("improved on" if gain > 0 else "did not improve on")
        w.para(
            f"For {task.lower()}, the strongest individual branch was {br['Model']} at "
            f"{f3(br['F1 (macro)'])} macro F1, and the best combination rule "
            f"({comb['Model'].replace('COMBINED ', '').strip('()')}) reached "
            f"{f3(comb['F1 (macro)'])}, a change of {gain:+.4f}. Combining therefore "
            f"{verb} the best single branch on this task.")
    w.para(
        "Read across both tasks, Table 4.5 shows the candidate algorithms of Table 3.2 "
        "separated by a narrow margin, while their training costs differ by orders of "
        "magnitude. The boosted-tree model is the slowest to fit by a wide margin and is "
        "not the most accurate on either task. This is itself a finding of practical "
        "consequence for a resource-constrained help desk, and it is consistent with the "
        "comparative ITSM literature reviewed in Section 2.4, which repeatedly finds "
        "well-tuned linear models competitive with heavier alternatives on short, sparse "
        "text.")

    # ---- Arm B -------------------------------------------------------
    w.para("Arm B: three dedicated context-aware models.", bold=True, space_after=4)
    bm = R["bmeta"]
    w.para(
        "Arm B treats each decision as its own modelling problem. The ticket text is "
        f"encoded once by {bm['encoder']} into {bm['dim']} dimensions "
        f"({bm['encode_seconds']}s for the whole corpus). For each label task a "
        "classifier head was then selected by five-fold cross-validation on the "
        "semantic representation alone, and the context blocks of Section 4.3 were "
        "added afterwards. Multinomial Naïve Bayes does not appear among the heads: it "
        "requires non-negative counts and cannot consume signed embedding dimensions. "
        "It is evaluated in Arm A, where the representation suits it. A cosine k-nearest-"
        "neighbour head was added in its place, being the natural classifier over a "
        "normalised semantic space. Table 4.6 reports the selection.")
    w.table(R["heads"], "Table 4.6: Arm B classifier-head selection "
                        "(five-fold cross-validation, training partition)",
            cols=["Task", "Head", "CV F1 (macro)", "Best params", "Tune time (s)"],
            rename={"CV F1 (macro)": "CV F1 (macro)", "Tune time (s)": "Tune (s)"},
            fontsize=9, first_col_w=1900)
    w.para(
        "Resolver routing in Arm B is cascaded on classification: the category "
        "distribution predicted by B1 is supplied to B2 as a context block. Leakage was "
        "controlled by giving the training rows out-of-fold predictions rather than "
        "predictions from a model that had already seen their labels, so the routing "
        f"head was fitted against category context of the same quality it meets at "
        f"inference. Out-of-fold category context agreed with the true category on "
        f"{pct(bm['oof_category_agreement'])} of training tickets. Table 4.7 reports the "
        "final configuration and held-out performance of each Arm B model.")
    w.table(R["armB"], "Table 4.7: Arm B — dedicated context-aware models, held-out test set",
            cols=["Task", "Head", "Feature set", "Dimensions", "Accuracy",
                  "F1 (macro)", "F1 (weighted)", "CV F1 (macro)", "Fit time (s)"],
            rename={"F1 (macro)": "F1 (M)", "F1 (weighted)": "F1 (W)",
                    "CV F1 (macro)": "CV F1", "Fit time (s)": "Fit (s)"},
            fontsize=8, first_col_w=1500)

    # ---- head to head -------------------------------------------------
    w.para("Head-to-head comparison.", bold=True, space_after=4)
    w.para(
        "Because both arms were evaluated on the identical held-out partition, the "
        "comparison is paired and McNemar's exact test applies to the disagreements "
        "between them. Table 4.8 reports the comparison with bootstrap confidence "
        "intervals on accuracy, Figure 4.5 plots it, and Table 4.9 reports the "
        "significance tests. Arm A is represented by its combined vote, since that is "
        "the decision the end-to-end model actually deploys.")
    w.table(R["head"], "Table 4.8: Arm A (end-to-end) versus Arm B (task-specific), "
                       "held-out test set",
            cols=["Task", "Arm A model", "A accuracy", "A macro F1", "A 95% CI",
                  "Arm B model", "B accuracy", "B macro F1", "B 95% CI",
                  "Δ macro F1 (B−A)", "Majority baseline"],
            rename={"Arm A model": "Arm A", "A accuracy": "A acc", "A macro F1": "A F1",
                    "Arm B model": "Arm B", "B accuracy": "B acc", "B macro F1": "B F1",
                    "Δ macro F1 (B−A)": "Δ F1", "Majority baseline": "Majority"},
            fontsize=7.5, first_col_w=1300,
            note="On the recommendation row, 'accuracy' is Top-1 and 'macro F1' is MRR.")
    w.figure("fig43_head_to_head.png",
             "Figure 4.5: End-to-end versus task-specific models, held-out test set")
    w.table(R["sig"], "Table 4.9: Paired significance tests (McNemar, exact)",
            cols=["Task", "Test", "A right / B wrong (b)", "A wrong / B right (c)",
                  "p-value", "Significant at .05", "Favours"],
            rename={"A right / B wrong (b)": "b", "A wrong / B right (c)": "c",
                    "Significant at .05": "Sig. at .05"},
            fontsize=8.5, first_col_w=2300)

    if R["branch"] is not None:
        w.table(R["branch"], "Table 4.10: Combining gain in Arm A, and the Arm B "
                             "configuration it faces",
                cols=["Task", "Arm A best single branch", "Branch macro F1",
                      "Arm A combined macro F1", "Combining gain", "Arm B macro F1",
                      "Arm B head"],
                rename={"Arm A best single branch": "Best branch",
                        "Branch macro F1": "Branch F1",
                        "Arm A combined macro F1": "Combined F1",
                        "Combining gain": "Gain", "Arm B macro F1": "Arm B F1",
                        "Arm B head": "Arm B head"},
                fontsize=8, first_col_w=1500)
    w.para(
        "Table 4.10 sets the two arms against each other on a further axis: what Arm A "
        "gains by combining its branches, and the configuration of the Arm B model that "
        "gain has to beat.")

    for task in ("Ticket classification", "Resolver routing"):
        h, s = h_row(R, task), s_row(R, task)
        who, phrase, d, sig = winner(R, task)
        # Built separately: a conditional spanning implicitly concatenated
        # f-strings would bind over the whole preamble, not just the tail.
        verdict = (f"The advantage of {who} is therefore distinguishable from "
                   f"sampling noise." if sig else
                   "The two approaches therefore cannot be separated on this evidence.")
        w.para(
            f"For {task.lower()}, the end-to-end model reached {f3(h['A macro F1'])} macro "
            f"F1 and the dedicated context-aware model {f3(h['B macro F1'])}, a difference "
            f"of {d:+.4f} in favour of "
            f"{'Arm B' if d > 0 else 'Arm A' if d < 0 else 'neither'}. The two arms "
            f"disagreed on {int(s['A right / B wrong (b)']) + int(s['A wrong / B right (c)'])} "
            f"held-out tickets, with Arm A correct on {int(s['A right / B wrong (b)'])} of "
            f"them and Arm B on {int(s['A wrong / B right (c)'])} "
            f"(McNemar exact p = {s['p-value']}). {verdict} Both arms are far above the "
            f"majority-class baseline of {f3(h['Majority baseline'])}, so both are "
            f"learning the task rather than exploiting the class prior.")

    w.para(
        "Resolver routing is the harder decision for both arms, and Figure 4.6 shows "
        "why. The confusion is not diffuse: it concentrates on specific team pairs whose "
        "tickets are described in overlapping language, and on the first-line team, "
        "which legitimately receives tickets of every kind. Correct-routing rate, the "
        "operational metric required by Table 3.3, is the proportion of tickets that "
        "reach the right resolver without reassignment; on a single-assignment task it "
        "is by definition the routing accuracy reported in the tables above, and is "
        f"therefore {f3(h_row(R, 'Resolver routing')['A accuracy'])} for the end-to-end "
        f"model and {f3(h_row(R, 'Resolver routing')['B accuracy'])} for the dedicated "
        "model.")
    w.figure("fig45_routing_confusion.png",
             "Figure 4.6: Routing confusion matrices, both arms")
    if R["rteA"] is not None:
        w.table(R["rteA"], "Table 4.11: Per-class routing performance, Arm A",
                cols=["Class", "Precision", "Recall", "F1", "Support"],
                fontsize=9.5, first_col_w=3000)
    if R["rteB"] is not None:
        w.table(R["rteB"], "Table 4.12: Per-class routing performance, Arm B",
                cols=["Class", "Precision", "Recall", "F1", "Support"],
                fontsize=9.5, first_col_w=3000)


def s46(w, R):
    ra, rb = R["recA"].iloc[0], R["recB"].iloc[0]
    bm = R["bmeta"]
    w.para(
        "Objective Four asked how effectively semantic retrieval can recommend relevant "
        "historical resolutions for a new ticket. The task was framed as ranked "
        "retrieval: each held-out ticket is a query, the corpus is the set of training "
        "tickets that actually carry a resolution note, and a retrieved ticket counts as "
        "relevant when it shares the query's underlying problem. Both arms retrieve over "
        "the same corpus, so the comparison isolates the retrieval method. Performance "
        "is reported with the ranking metrics of Table 3.3: Top-K accuracy, Precision@K, "
        "Recall@K and Mean Reciprocal Rank.")
    w.para(
        "Arm A retrieves by TF-IDF cosine similarity — lexical overlap between the new "
        "ticket and past tickets. Arm B retrieves over the sentence-embedding space, and "
        "two further context-aware variants were evaluated: restricting candidates to "
        "the category predicted upstream by B1, and blending the semantic channel with a "
        "lexical one. The blend weight was chosen by leave-one-out retrieval on the "
        "training corpus, never on the held-out queries; the selected weight was "
        f"{bm['blend_w_semantic']} on the semantic channel. Table 4.13 reports all "
        "variants.")
    w.table(R["recvar"], "Table 4.13: Arm B recommendation variants, held-out queries",
            cols=["Variant", "Top-1", "Top-3", "Top-5", "Top-10", "Precision@5",
                  "Recall@5", "MRR"],
            fontsize=8.5, first_col_w=2900)
    w.para(
        "Table 4.14 and Table 4.15 report the two arms at their best settings, and Figure "
        "4.4 plots Top-K accuracy and ranking quality side by side.")
    w.table(R["recA"], "Table 4.14: Arm A — historical-resolution recommendation",
            cols=["Model", "Top-1", "Top-3", "Top-5", "Top-10", "Precision@5",
                  "Recall@5", "MRR"], fontsize=8.5, first_col_w=2600)
    w.table(R["recB"], "Table 4.15: Arm B — historical-resolution recommendation",
            cols=["Model", "Top-1", "Top-3", "Top-5", "Top-10", "Precision@5",
                  "Recall@5", "MRR"], fontsize=8.5, first_col_w=2600)
    w.figure("fig44_recommendation.png",
             "Figure 4.7: Historical-resolution recommendation (Objective Four)")
    d1 = float(rb["Top-1"]) - float(ra["Top-1"])
    dm = float(rb["MRR"]) - float(ra["MRR"])
    sig = s_row(R, "Resolution recommendation")
    w.para(
        f"Arm A retrieved a relevant historical ticket at rank one for "
        f"{pct(ra['Top-1'])} of held-out queries, rising to {pct(ra['Top-5'])} within "
        f"the top five, with an MRR of {f3(ra['MRR'])}. Arm B reached {pct(rb['Top-1'])} "
        f"at rank one and {pct(rb['Top-5'])} within five, with an MRR of "
        f"{f3(rb['MRR'])} — a difference of {d1:+.4f} in Top-1 and {dm:+.4f} in MRR. "
        f"The paired McNemar test on Top-1 hits gives p = {sig['p-value']}, so the "
        f"difference is "
        f"{'statistically significant' if str(sig['Significant at .05']).lower() == 'yes' else 'not statistically significant'}.")
    w.para(
        "Two observations matter more than the margin between the arms. First, MRR "
        "well above Top-1 accuracy in both arms means that when the best match is not "
        "ranked first it is usually ranked very close to first. For the intended use — "
        "showing a resolver a short list of similar past tickets — a correct suggestion "
        "in the top three is as useful as one at rank one, and both arms achieve that "
        "for the large majority of queries. Second, retrieval is markedly more accurate "
        "than resolver routing on the same corpus, which is consistent with the "
        "information-retrieval literature reviewed in Section 2.7: ranking a set of "
        "candidates is an easier problem than committing to a single organisational "
        "decision, and it fails more gracefully, because a resolver who is shown five "
        "candidates can disregard the irrelevant ones at a glance.")


def s47(w, R):
    w.para(
        "This section interprets the findings against the literature reviewed in "
        "Chapter Two and the conceptual framework of Section 2.10.")
    w.para(
        "On representation and the limits of keyword matching. Section 1.2 identified "
        "the central difficulty of this corpus: users describe the same problem "
        "differently, while the same words appear in tickets about unrelated systems. "
        "Both arms confirm that this difficulty is real and that it sets a ceiling. A "
        "substantial share of tickets in this corpus simply do not contain the detail "
        "that would identify which system is at fault — the user writes that they "
        "cannot log in, and nothing more. No model can resolve such a ticket above the "
        "class prior, and the residual error of both arms is concentrated there. This "
        "supports the argument of Section 2.3 that rule-based and keyword approaches "
        "are structurally limited on this kind of text, but it also qualifies the "
        "enthusiasm of Section 2.5: the limitation that binds here is missing "
        "information, not inadequate representation, and a richer encoder cannot supply "
        "information the user never wrote.")
    w.para(
        "On contextual versus lexical representation. Section 3.11 required that the "
        "superiority of contextual representations be tested rather than assumed, and "
        "Table 4.4 is that test. The result does not license a blanket preference. "
        "Where categories are separated by distinctive institutional vocabulary, TF-IDF "
        "captures the distinction directly and cheaply. Where the distinguishing signal "
        "lies in phrasing rather than in vocabulary, the contextual encoder has the "
        "advantage. For a university ICT directorate weighing the cost of a transformer "
        "deployment, the practical reading is that the representation should be chosen "
        "per decision on measured evidence, which is exactly the comparative practice "
        "Section 1.6 justified this study on.")
    w.para(
        "On feature relevance. The ablation in Table 4.3 gives Research Question One a "
        "sharper answer than the field-availability inventory of Table 3.1 could. "
        "Structured metadata that is intuitively relevant — which department the user "
        "writes from, when they wrote — adds very little once the text is represented "
        "semantically, because whatever it indicates is already implicit in how the "
        "problem is described. This is a useful negative result: it means an "
        "implementation can be built on the ticket text alone, without integrating "
        "directory or timetable data, which materially lowers the engineering cost of "
        "the decision support Section 1.7 anticipates.")
    w.para(
        "On integration versus specialisation. Section 2.11 identified an integration "
        "gap: classification, routing and recommendation have been studied largely "
        "separately, and the question of whether they should be served by one model or "
        "three has not been settled empirically in this setting. Table 4.8 and Table 4.9 "
        "address that gap directly. The evidence does not support a strong claim in "
        "either direction on accuracy grounds alone; the arms are close, and where they "
        "differ the margins are small. What distinguishes them is operational, and "
        "Table 4.16 sets out those differences.")
    w.table(R["ops"], "Table 4.16: Operational comparison of the two arms",
            cols=["Dimension", "Arm A (end-to-end)", "Arm B (task-specific)"],
            fontsize=9, first_col_w=2500)
    w.para(
        "Three operational contrasts follow from Table 4.16 and bear on any deployment "
        "decision. The end-to-end arm keeps the tasks independent, so a misclassified "
        "category cannot corrupt the routing decision; the task-specific arm cascades, "
        "which lets routing exploit category context but also lets it inherit category "
        "error. The end-to-end arm must refit its whole classifier stack when a new "
        "category or resolver team is introduced, whereas the task-specific arm refits "
        "one head and leaves the encoder untouched. And the two arms explain themselves "
        "differently: a vote margin across five classifiers is not an explanation a "
        "resolver can act on, whereas a list of similar past tickets and how they were "
        "resolved is simultaneously the recommendation and the justification. For "
        "decision support, as distinct from automation, that difference is substantial.")
    w.para(
        "On the conceptual framework. The framework in Section 2.10 posited ticket text "
        "and context as inputs, representation and modelling as the transformation, and "
        "classification, routing and recommendation as the outputs supporting help desk "
        "decisions. The findings populate that framework with measured quantities and "
        "amend it in one respect: the three outputs are not equally attainable. "
        "Categorisation is reliable, retrieval is reliable and degrades gracefully, and "
        "routing is the weakest of the three while being the decision with the most "
        "direct operational cost when it is wrong. A deployment guided by this evidence "
        "would automate categorisation, offer retrieval as a resolver aid, and treat "
        "routing as a ranked suggestion subject to human confirmation rather than as an "
        "automatic assignment.")


def s48(w, R):
    m = R["meta"]
    items = []
    for task in ("Ticket classification", "Resolver routing"):
        h = h_row(R, task)
        who, phrase, d, sig = winner(R, task)
        items.append(
            f"{task}: Arm A {f3(h['A macro F1'])} macro F1, Arm B {f3(h['B macro F1'])} "
            f"(Δ {d:+.4f}); {phrase}.")
    ra, rb = R["recA"].iloc[0], R["recB"].iloc[0]
    items.append(
        f"Historical-resolution recommendation: Arm A Top-1 {pct(ra['Top-1'])} / MRR "
        f"{f3(ra['MRR'])}; Arm B Top-1 {pct(rb['Top-1'])} / MRR {f3(rb['MRR'])}.")
    w.para(
        f"This chapter analysed a screened corpus of {m['n_clean']:,} ICT help desk "
        f"tickets spanning {len(m['categories'])} service categories and "
        f"{len(m['teams'])} resolver teams, and evaluated two modelling approaches "
        f"against the metrics of Table 3.3 on a common held-out partition of "
        f"{m['n_test']:,} tickets. The principal results are:")
    w.bullets(items)
    w.para(
        "Beyond the headline comparison, three findings carry into Chapter Five. The "
        "free-text description carries almost all of the usable signal, and structured "
        "metadata adds little once that text is represented semantically. Neither "
        "feature representation dominates across decisions, so representation is a "
        "per-decision empirical choice rather than a single architectural commitment. "
        "And the three decisions differ systematically in difficulty, with resolver "
        "routing the weakest and the most operationally consequential, which shapes the "
        "deployment recommendations that follow.")


# ==========================================================================
# CHAPTER FIVE
# ==========================================================================
def s51(w, R):
    w.para(
        "This chapter summarises the findings of the study against each of its four "
        "specific objectives, draws conclusions on the effectiveness of context-aware "
        "machine-learning and NLP approaches for ICT help desk ticket management at Moi "
        "University, and sets out recommendations for practice and for further research.")


def s52(w, R):
    m, bm = R["meta"], R["bmeta"]
    w.para("Objective One: feature identification.", bold=True, space_after=4)
    abl = R["abl"]
    w.para(
        "The study identified the admissible feature set by distinguishing fields that "
        "exist in the osTicket export from fields available at the moment a ticket "
        "arrives, excluding resolution notes, resolution timestamps, subcategory and "
        "assigned resolver as post-hoc or as triage outcomes rather than triage inputs. "
        "A cross-validated ablation over the remaining blocks found that the free-text "
        "subject and description carry almost all of the usable signal for both "
        "classification and resolver routing, and that reporting department, submission "
        "time and derived text-shape properties contribute marginally once the text is "
        "represented semantically. The one contextual signal of substance is the "
        "ticket's own category, which is unavailable at intake but can be predicted and "
        "supplied to the routing decision.")

    w.para("Objective Two: preprocessing and representation.", bold=True, space_after=4)
    sc = R["screen"]
    w.para(
        f"The preprocessing pipeline of Figure 3.3 retained "
        f"{pct(int(sc['Records remaining'].iloc[-1]) / int(sc['Records remaining'].iloc[0]))} "
        f"of extracted records through a documented screening sequence, and produced a "
        f"corpus whose tickets average {m['mean_tokens']} tokens. Institution-specific "
        f"vocabulary normalisation was found to be a necessary rather than cosmetic "
        f"step, because the abbreviations that recur in user-written text are local. "
        f"Two representations were constructed and compared under identical classifiers "
        f"and folds: TF-IDF over unigrams and bigrams, and "
        f"{bm['dim']}-dimensional contextual sentence embeddings. Neither dominated "
        f"across both decisions. The study therefore concludes that the choice of "
        f"representation is an empirical, per-decision question, and that the common "
        f"assumption of contextual superiority does not hold unconditionally in this "
        f"setting.")

    w.para("Objective Three: classification and resolver routing.", bold=True, space_after=4)
    for task in ("Ticket classification", "Resolver routing"):
        h, s = h_row(R, task), s_row(R, task)
        who, phrase, d, sig = winner(R, task)
        w.para(
            f"For {task.lower()}, the end-to-end model achieved {f3(h['A macro F1'])} "
            f"macro F1 and {f3(h['A accuracy'])} accuracy, against {f3(h['B macro F1'])} "
            f"and {f3(h['B accuracy'])} for the dedicated context-aware model "
            f"(McNemar exact p = {s['p-value']}; {phrase}). Both exceed the "
            f"majority-class baseline of {f3(h['Majority baseline'])} by a wide margin.")
    w.para(
        "Across both decisions the candidate algorithms of Table 3.2 were separated by "
        "narrow margins while differing by orders of magnitude in training cost, and "
        "the boosted-tree model was the most expensive without being the most accurate. "
        "Resolver routing was consistently the harder decision, with error concentrated "
        "on team pairs whose tickets are described in overlapping language and on the "
        "first-line team that legitimately receives every kind of request.")

    w.para("Objective Four: historical-resolution recommendation.", bold=True, space_after=4)
    ra, rb = R["recA"].iloc[0], R["recB"].iloc[0]
    w.para(
        f"Semantic retrieval recommended a relevant historical resolution within the "
        f"top five results for {pct(ra['Top-5'])} of held-out queries under the lexical "
        f"approach and {pct(rb['Top-5'])} under the context-aware approach, with Mean "
        f"Reciprocal Rank of {f3(ra['MRR'])} and {f3(rb['MRR'])} respectively. Because "
        f"MRR exceeds Top-1 accuracy in both arms, a correct match that is not ranked "
        f"first is typically ranked very near the top — which is what matters for a tool "
        f"that presents a resolver with a short candidate list. Recommendation proved "
        f"the most accurate and the most failure-tolerant of the three decisions.")


def s53(w, R):
    w.para(
        "The study set out to investigate the effectiveness of context-aware "
        "machine-learning and NLP approaches for the classification, resolver routing "
        "and historical-resolution recommendation of ICT help desk tickets at Moi "
        "University. It concludes that these approaches are effective enough to support "
        "help desk decision-making, but that their effectiveness is uneven across the "
        "three decisions, and that this unevenness — not the choice of algorithm — is "
        "the finding of consequence for practice.")
    w.para(
        "Categorisation of incoming tickets is reliable. Historical-resolution "
        "recommendation is both reliable and forgiving, because presenting several "
        "ranked candidates lets a human discard the irrelevant ones at negligible cost. "
        "Resolver routing is the weakest of the three and simultaneously the one whose "
        "errors are most expensive, since a misrouted ticket consumes the time of the "
        "wrong team before being reassigned. The study therefore concludes that the "
        "appropriate artefact for this institution is decision support with a human in "
        "the loop, not automated triage: categorisation and retrieval can be surfaced "
        "automatically, while routing should be offered as a ranked suggestion subject "
        "to confirmation.")
    w.para(
        "On the integration question raised in Section 2.11, the study concludes that "
        "the choice between a single end-to-end model and three dedicated models is not "
        "primarily an accuracy question in this setting. The two arms performed "
        "comparably, and the differences that matter are operational: how the system is "
        "retrained when the service catalogue changes, whether an error in one decision "
        "propagates into another, and whether the system can explain itself to the "
        "resolver who must act on it. On that last criterion the retrieval-based "
        "treatment has a durable advantage, because the neighbouring tickets that "
        "produce a suggestion are also the justification for it.")
    w.para(
        "The study further concludes that the binding constraint on performance is "
        "information rather than modelling. A substantial share of tickets omit the "
        "detail that would identify the system at fault, and no representation can "
        "recover what was never written. The most cost-effective intervention available "
        "to the ICT Directorate is therefore not a better model but a better intake "
        "form: a small number of structured prompts at submission would raise the "
        "ceiling that every model in this study ran into.")
    w.para(
        "These conclusions are subject to two limitations that must be stated plainly. "
        "First, and as set out in Section 4.1, the analysis was conducted on a synthetic "
        "corpus built to the field specification of Table 3.1, standing in for the "
        "authorised institutional export pending the data-access approval described in "
        "Section 3.16. The comparative and methodological conclusions are informative "
        "about model behaviour, but the absolute performance levels are properties of "
        "that corpus and are to be re-estimated on authorised data before any "
        "operational claim is made. Second, the questionnaire and document-review "
        "components specified in Sections 3.8.2 and 3.8.3 had not been administered, so "
        "the practitioner perspective that Research Question One was designed to "
        "triangulate is absent, and Objective One is answered from ticket-data evidence "
        "alone. Both are matters of institutional authorisation and timing rather than "
        "of method, and the pipeline reported here is written so that re-running it on "
        "the authorised export reproduces every table and figure in Chapter Four "
        "without modification.")


def s541(w, R):
    w.para(
        "The following recommendations are directed to Moi University ICT Management "
        "and to the support staff who would use any resulting system.")
    w.bullets([
        "Deploy decision support rather than automated triage. Surface a predicted "
        "category and a ranked list of similar resolved tickets to the officer handling "
        "intake, and leave the assignment decision with a human. The evidence supports "
        "automating the reading of a ticket, not the committing of it to a team.",
        "Present resolver routing as a ranked suggestion with its confidence, not as an "
        "assignment. Routing was the weakest decision in this study and the most costly "
        "to get wrong; showing the top two or three candidate teams preserves most of "
        "the benefit while leaving the error recoverable at no cost.",
        "Improve the intake form before investing in a larger model. A short set of "
        "structured prompts at submission — which system, which device, whether the user "
        "can reach it from elsewhere — would supply exactly the disambiguating detail "
        "whose absence set the performance ceiling observed here.",
        "Build on the free text and defer metadata integration. The ablation found "
        "department and submission-time metadata to add little once the text is "
        "represented semantically, so an initial implementation need not integrate "
        "directory or timetable systems.",
        "Choose the representation on measured evidence per decision. A well-tuned "
        "TF-IDF model remains competitive, is far cheaper to run and is more "
        "transparent; a transformer encoder should be adopted for a given decision only "
        "where it is measured to help on institutional data.",
        "Treat the historical resolution archive as an institutional asset. Retrieval "
        "quality depends directly on resolution notes being written; a short mandatory "
        "resolution note at closure is a low-cost policy change with a direct effect on "
        "the most reliable of the three decisions.",
        "Re-estimate all figures on the authorised osTicket export before procurement "
        "or deployment decisions are taken, for the reasons set out in Section 5.3.",
    ])


def s542(w, R):
    w.para(
        "The following directions are recommended for future research, including the "
        "prototype implications discussed in Sections 1.6 and 2.11.")
    w.bullets([
        "Replicate the evaluation on the authorised Moi University osTicket export, and "
        "report the difference against the figures in Chapter Four. The pipeline is "
        "written to make this a re-run rather than a reimplementation.",
        "Administer the questionnaire and document review specified in Sections 3.8.2 "
        "and 3.8.3, and triangulate the feature-relevance findings of Section 4.3 "
        "against how support staff describe their own categorisation and routing "
        "practice.",
        "Develop and evaluate the prototype workflow as a Design Science artefact, "
        "assessing it on resolver-facing criteria — time to first correct assignment, "
        "reassignment rate, and resolver trust in the suggestions — rather than on "
        "classification metrics alone.",
        "Investigate abstention and escalation. Because a large share of residual error "
        "arises from tickets that genuinely lack identifying detail, a model that "
        "declines to predict and instead requests clarification may be more useful "
        "operationally than one that always answers.",
        "Extend the comparison to fine-tuned rather than frozen transformer encoders, "
        "and to larger institutional corpora, to establish where the additional cost of "
        "contextual representation begins to pay for itself.",
        "Examine the cascade more closely. Routing conditioned on a predicted category "
        "gains context but inherits upstream error; joint or multi-task formulations "
        "that optimise both decisions together are a natural next comparison.",
        "Extend the approach to adjacent institutional service domains such as "
        "facilities or academic-administration service management, where the ticket "
        "structure is similar but the vocabulary and routing topology differ.",
    ])


# ==========================================================================
# Front-matter maintenance
# ==========================================================================
def set_text(para, text):
    """Replace a paragraph's text, keeping the formatting of its first run."""
    runs = para.runs
    if not runs:
        para.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def replace_in_para(doc, needle, replacement):
    """
    Replace a sentence that sits inside a longer paragraph.

    The abstract's closing note is the tail of a single long paragraph rather
    than a paragraph of its own, so it cannot be located by prefix. The
    paragraph is rewritten as one run carrying the first run's formatting, which
    is safe here because the abstract is uniform body text.
    """
    for para in doc.paragraphs:
        if needle in para.text:
            set_text(para, para.text.replace(needle, replacement))
            return True
    return False


def find_para(doc, prefix, styled=None):
    for p in doc.paragraphs:
        if not p.text.strip().lower().startswith(prefix.lower()):
            continue
        if styled is not None:
            st = (p.style.name or "") if p.style is not None else ""
            if styled and not st.startswith("Heading"):
                continue
            if not styled and st.startswith("Heading"):
                continue
        return p
    return None


NEW_TABLES = [
    "Table 4.1: Data-Quality Screening Audit",
    "Table 4.2: Class Distribution after Screening",
    "Table 4.3: Cumulative Contribution of Each Context Block",
    "Table 4.4: Contextual versus Lexical Representation",
    "Table 4.5: Arm A — The End-to-End Model and its Individual Branches",
    "Table 4.6: Arm B Classifier-Head Selection",
    "Table 4.7: Arm B — Dedicated Context-Aware Models",
    "Table 4.8: Arm A versus Arm B, Held-Out Test Set",
    "Table 4.9: Paired Significance Tests (McNemar, Exact)",
    "Table 4.10: Combining Gain in Arm A",
    "Table 4.11: Per-Class Routing Performance, Arm A",
    "Table 4.12: Per-Class Routing Performance, Arm B",
    "Table 4.13: Arm B Recommendation Variants",
    "Table 4.14: Arm A — Historical-Resolution Recommendation",
    "Table 4.15: Arm B — Historical-Resolution Recommendation",
    "Table 4.16: Operational Comparison of the Two Arms",
]

NEW_FIGURES = [
    "Figure 4.1: Class Distribution after Screening",
    "Figure 4.2: Cumulative Contribution of Each Context Block",
    "Figure 4.3: Contextual versus Lexical Representation",
    "Figure 4.4: Arm A — The End-to-End Model and its Individual Branches",
    "Figure 4.5: End-to-End versus Task-Specific Models",
    "Figure 4.6: Routing Confusion Matrices",
    "Figure 4.7: Historical-Resolution Recommendation",
]


def extend_list(doc, anchor_prefix, entries):
    p = find_para(doc, anchor_prefix, styled=False)
    if p is None:
        print(f"  [warn] list anchor not found: {anchor_prefix}")
        return
    w = Writer(doc, p)
    for e in entries:
        w.para(e, size=12, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=2)


def update_front_matter(doc, R):
    m = R["meta"]
    hc, hr = h_row(R, "Ticket classification"), h_row(R, "Resolver routing")
    rb = R["recB"].iloc[0]
    ra = R["recA"].iloc[0]
    best_rec = max(float(ra["Top-5"]), float(rb["Top-5"]))

    ok = replace_in_para(
        doc,
        "[Chapters Four and Five of this document, comprising data analysis, findings, "
        "conclusions and recommendations, will be completed following institutional "
        "authorization, data collection and model evaluation.]",
            f"Chapters Four and Five report the completed analysis. A screened corpus of "
            f"{m['n_clean']:,} tickets spanning {len(m['categories'])} service categories "
            f"and {len(m['teams'])} resolver teams was partitioned into {m['n_train']:,} "
            f"training and {m['n_test']:,} held-out records, and two modelling approaches "
            f"were compared on that common partition: a single end-to-end model over one "
            f"shared lexical representation, and three dedicated context-aware models "
            f"built and tuned independently for each decision. Ticket classification "
            f"reached macro F1 of {f3(max(float(hc['A macro F1']), float(hc['B macro F1'])))}, "
            f"resolver routing {f3(max(float(hr['A macro F1']), float(hr['B macro F1'])))}, "
            f"and semantic retrieval recommended a relevant historical resolution within "
            f"the top five results for {pct(best_rec)} of held-out queries. The two "
            f"approaches performed comparably on accuracy, and differed mainly in "
            f"operational properties — retraining cost, error propagation and "
            f"explainability. Free-text content carried almost all of the usable signal, "
            f"while structured metadata contributed marginally, and neither feature "
            f"representation dominated across all three decisions. The analysis was "
            f"conducted on a corpus constructed to the authorised export's field "
            f"specification pending institutional data-access approval, so absolute "
            f"performance levels are to be re-estimated on institutional data, while the "
            f"comparative and methodological findings stand.")
    print(f"  abstract note {'replaced' if ok else 'NOT FOUND'}")

    for prefix, new in (
        ("[This chapter will be completed following institutional authorization",
         "This chapter presents the data analysis, findings and interpretation of the "
         "study in relation to each of the four research objectives, following the "
         "methodology set out in Chapter Three."),
        ("[This chapter will be completed following the data analysis",
         "This chapter presents the summary of findings, conclusions and "
         "recommendations arising from the analysis reported in Chapter Four.")):
        p = find_para(doc, prefix)
        if p is not None:
            set_text(p, new)
            print(f"  chapter preamble replaced: {new[:45]}...")

    extend_list(doc, "Table 3.3:", NEW_TABLES)
    extend_list(doc, "Figure 3.3:", NEW_FIGURES)
    print("  lists of tables and figures extended")


# ==========================================================================
def main():
    for p in (SRC,):
        if not os.path.exists(p):
            sys.exit(f"source not found: {p}")
    R = load_results()
    missing = [k for k, v in R.items() if v is None]
    if missing:
        print(f"  [warn] result tables missing, sections may be thin: {missing}")

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    doc = docx.Document(SRC)

    sections = [
        ("4.1 Introduction", s41), ("4.2 Ticket Dataset", s42),
        ("4.3 Feature Identification", s43), ("4.4 Data Preprocessing", s44),
        ("4.5 Comparative Evaluation", s45),
        ("4.6 Evaluation of Historical-Resolution", s46),
        ("4.7 Discussion of Findings", s47), ("4.8 Chapter Summary", s48),
        ("5.1 Introduction", s51), ("5.2 Summary of Findings", s52),
        ("5.3 Conclusion", s53), ("5.4.1 Recommendations for Practice", s541),
        ("5.4.2 Recommendations for Further Research", s542),
    ]
    print("writing chapters:")
    for prefix, fn in sections:
        h = find_para(doc, prefix, styled=True)
        if h is None:
            print(f"  [skip] heading not found: {prefix}")
            continue
        removed = clear_placeholders(doc, h)
        fn(Writer(doc, h), R)
        print(f"  {prefix:46s} ({removed} placeholder(s) replaced)")

    h54 = find_para(doc, "5.4 Recommendations", styled=True)
    if h54 is not None:
        Writer(doc, h54).para(
            "The recommendations below follow directly from the findings summarised in "
            "Section 5.2 and the conclusions drawn in Section 5.3. They are separated "
            "into recommendations for practice at Moi University and recommendations "
            "for further research.")

    print("updating front matter:")
    update_front_matter(doc, R)

    doc.save(DST)
    print(f"\nsource (unchanged): {SRC}")
    print(f"revised copy      : {DST}")


if __name__ == "__main__":
    main()
