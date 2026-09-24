"""
Text cleaning and construction -- the single definition used by BOTH training
and serving.

This module exists so that a ticket typed into the prototype is put through
exactly the same transformation as every ticket the models were trained on.
If cleaning lived only in the preprocessing script, the served model would
silently see differently-shaped text than it was fitted on -- train/serve skew,
which is invisible in the metrics and wrong in the demo.
"""
import re


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



def build_text(subject, description):
    """
    Construct the model input from a subject and a description.

    The subject is a user-written summary and is short, whereas the description
    may ramble. Concatenating them directly lets a long description swamp the
    summary, so the subject is repeated once before the description -- weighting
    it without discarding anything. This is a representation choice, reported in
    Section 4.4 of the thesis.
    """
    s = clean_text(subject)
    d = clean_text(description)
    return f"{s} {s} {d}".strip()
