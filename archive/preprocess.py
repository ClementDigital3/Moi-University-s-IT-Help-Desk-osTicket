"""
Data processing and feature preparation -- implements Section 3.11 of the thesis.

Pipeline stages, in the order given in Figure 3.3:
    quality screening -> missing-value handling -> text cleaning ->
    tokenization / normalization -> label encoding -> class-imbalance assessment

Outputs
    data/processed/clean.csv          analysable corpus
    data/processed/split_train.csv    stratified training partition
    data/processed/split_test.csv     held-out test partition
    results/tables/t41_screening.csv  screening audit -> Chapter 4, Sec. 4.4
    results/tables/t42_classdist.csv  class distributions -> Chapter 4, Sec. 4.4

The train/test split is written to disk and reused by BOTH model arms so that
every comparison is paired on identical data (required for the McNemar tests).
"""
import os, re, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
TABLES = os.path.join(ROOT, "results", "tables")
SEED = 42
TEST_SIZE = 0.20

# Domain abbreviation normalisation. Help desk text is written by end users, so
# these expansions are institution-specific vocabulary repair, not generic NLP.
NORMALISE = {
    r"\bpls\b": "please", r"\bpwd\b": "password", r"\bcant\b": "cannot",
    r"\bcan not\b": "cannot", r"\bdept\b": "department", r"\buni\b": "university",
    r"\bcomp\b": "computer", r"\binfo\b": "information", r"\bthanx\b": "thanks",
    r"\bwifi\b": "wifi", r"\bwi-fi\b": "wifi", r"\bwi fi\b": "wifi",
    r"\be-?learning\b": "elearning", r"\blms\b": "elearning",
    r"\bu\b": "you", r"\burgntly\b": "urgently", r"\bplatfom\b": "platform",
    r"\bsis\b": "student information system", r"\bmis\b": "student information system",
}

TEST_TICKET = re.compile(r"^\s*(test(ing)?( ticket)?( please ignore)?|xxx|ignore this)\s*$", re.I)
JUNK_DESC = re.compile(r"^\s*(as above|see subject|-|\.|n/?a)?\s*$", re.I)


def clean_text(s: str) -> str:
    """Lowercase, strip artefacts, normalise domain abbreviations, squeeze space."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)          # urls
    s = re.sub(r"\S+@\S+\.\S+", " ", s)                    # emails
    s = re.sub(r"\bmu-\d+\b", " ", s)                      # ticket references
    s = re.sub(r"\d+", " ", s)                             # digits carry no class signal here
    for pat, rep in NORMALISE.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def screen(df: pd.DataFrame):
    """Quality screening. Returns (kept_df, audit_rows)."""
    audit = []
    n0 = len(df)

    def drop(mask, reason):
        nonlocal df
        k = int(mask.sum())
        if k:
            df = df[~mask].copy()
        audit.append({"Screening step": reason, "Records removed": k,
                      "Records remaining": len(df)})

    audit.append({"Screening step": "Raw records extracted", "Records removed": 0,
                  "Records remaining": n0})

    subj = df["Subject"].fillna("")
    desc = df["Description"].fillna("")
    drop(subj.str.match(TEST_TICKET) | desc.str.match(TEST_TICKET),
         "Test / placeholder tickets removed")
    drop(df.duplicated(subset=["Subject", "Description", "Category", "Created"], keep="first"),
         "Exact duplicate submissions removed")
    drop(df["Description"].fillna("").apply(lambda s: bool(JUNK_DESC.match(s)))
         & (df["Subject"].fillna("").str.len() < 12),
         "Records with no usable free text removed")
    drop(df["Category"].fillna("").str.strip() == "",
         "Records with missing category label removed")
    drop(df["Assigned Team"].fillna("").str.strip() == "",
         "Records with no assigned resolver removed")
    return df, audit


def main():
    os.makedirs(PROC, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)

    df = pd.read_csv(os.path.join(RAW, "osticket_export.csv"), dtype=str).fillna("")
    gt = pd.read_csv(os.path.join(RAW, "ground_truth_problem_ids.csv"), dtype=str)
    df = df.merge(gt, on="Ticket ID", how="left")

    df, audit = screen(df)
    audit_df = pd.DataFrame(audit)
    audit_df["Retention %"] = (100 * audit_df["Records remaining"] / audit[0]["Records remaining"]).round(1)
    audit_df.to_csv(os.path.join(TABLES, "t41_screening.csv"), index=False)
    print("--- Screening audit (Table 4.1) ---")
    print(audit_df.to_string(index=False))

    # ---- text construction -------------------------------------------------
    # The subject is a user-written summary and is repeated once so that it is
    # not swamped by a long body; this is reported as a representation choice.
    df["subject_clean"] = df["Subject"].apply(clean_text)
    df["desc_clean"] = df["Description"].apply(clean_text)
    df["text"] = (df["subject_clean"] + " " + df["subject_clean"] + " " + df["desc_clean"]).str.strip()
    df["res_clean"] = df["Resolution Notes"].fillna("")

    df = df[df["text"].str.split().str.len() >= 3].copy()

    # ---- label encoding ----------------------------------------------------
    df["y_category"] = df["Category"].astype("category")
    df["y_team"] = df["Assigned Team"].astype("category")
    cat_map = dict(enumerate(df["y_category"].cat.categories))
    team_map = dict(enumerate(df["y_team"].cat.categories))
    df["y_category_id"] = df["y_category"].cat.codes
    df["y_team_id"] = df["y_team"].cat.codes

    # ---- class-imbalance assessment ---------------------------------------
    rows = []
    for label, col in (("Category", "Category"), ("Resolver team", "Assigned Team")):
        vc = df[col].value_counts()
        for k, v in vc.items():
            rows.append({"Label type": label, "Class": k, "n": int(v),
                         "% of corpus": round(100 * v / len(df), 1)})
        rows.append({"Label type": label, "Class": "-- imbalance ratio --",
                     "n": round(vc.max() / vc.min(), 2), "% of corpus": ""})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES, "t42_classdist.csv"), index=False)

    # ---- stratified split, shared by both arms ----------------------------
    tr, te = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                              stratify=df["Category"])
    tr.to_csv(os.path.join(PROC, "split_train.csv"), index=False)
    te.to_csv(os.path.join(PROC, "split_test.csv"), index=False)
    df.to_csv(os.path.join(PROC, "clean.csv"), index=False)

    meta = {"n_clean": len(df), "n_train": len(tr), "n_test": len(te),
            "seed": SEED, "test_size": TEST_SIZE,
            "categories": list(cat_map.values()), "teams": list(team_map.values()),
            "mean_tokens": round(float(df["text"].str.split().str.len().mean()), 1),
            "median_tokens": int(df["text"].str.split().str.len().median())}
    with open(os.path.join(PROC, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nclean={len(df)}  train={len(tr)}  test={len(te)}")
    print(f"mean tokens/ticket={meta['mean_tokens']}  median={meta['median_tokens']}")
    print(f"categories={len(cat_map)}  teams={len(team_map)}")


if __name__ == "__main__":
    main()
