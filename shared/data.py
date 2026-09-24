"""Loading the analysable corpus and the two text views the models use."""
import json, os
import pandas as pd

from .paths import DATA_PROC


def load_splits():
    """
    The stratified train/test partition written once by analysis/preprocess.py.

    Every model loads the identical partition from disk rather than re-splitting,
    which is what makes the paired McNemar comparisons in analysis/compare.py
    admissible.
    """
    tr = pd.read_csv(os.path.join(DATA_PROC, "split_train.csv"), dtype=str).fillna("")
    te = pd.read_csv(os.path.join(DATA_PROC, "split_test.csv"), dtype=str).fillna("")
    meta = json.load(open(os.path.join(DATA_PROC, "meta.json")))
    return tr, te, meta


def clean_text(df):
    """
    The cleaned, normalised view: lowercased, stripped of URLs, emails, ticket
    references and digits, with institutional abbreviations expanded.

    This is what the lexical (TF-IDF) models consume, because TF-IDF gains
    nothing from casing or punctuation.
    """
    return df["text"]


def raw_text(df):
    """
    The natural, uncleaned view: subject and description as the user wrote them.

    This is what the contextual sentence encoder consumes, because sentence
    boundaries and casing carry contextual signal that cleaning destroys. The
    text handed to each model is part of the representation under test, not an
    inconsistency between models.
    """
    subj = df["Subject"].fillna("").astype(str).str.strip()
    desc = df["Description"].fillna("").astype(str).str.strip()
    return (subj + ". " + desc).str.replace(r"\s+", " ", regex=True).str.strip()
