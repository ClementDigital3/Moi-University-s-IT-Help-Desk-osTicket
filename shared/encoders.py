"""
Text representations.

Two families, as Section 3.11 of the thesis requires:

    contextual  a sentence-transformer encoder (the "contextual / semantic
                benchmark" of Table 3.2)
    lexical     TF-IDF, optionally reduced to dense latent-semantic vectors

Section 3.11 requires that the superiority of the contextual representation be
tested rather than assumed, so both are built here and compared in
analysis/compare.py rather than one being adopted by default.

The contextual encode pass is the single most expensive step in the whole
study, so it is cached on disk and shared by every model that needs it.
"""
import os, json, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

from .paths import CACHE, SEED
from .data import raw_text

ENCODER_CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
]


def tfidf_vectorizer():
    """The lexical representation, shared by the end-to-end model and retrieval."""
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.85,
                           sublinear_tf=True, max_features=30000,
                           strip_accents="unicode")


def encode_sbert(train_texts, test_texts):
    from sentence_transformers import SentenceTransformer
    last = None
    for name in ENCODER_CANDIDATES:
        try:
            m = SentenceTransformer(name)
        except Exception as e:                       # offline / not cached
            last = e
            print(f"  [encoder] {name} unavailable ({type(e).__name__})")
            continue
        enc = lambda xs: np.asarray(m.encode(list(xs), batch_size=64,
                                             show_progress_bar=False,
                                             normalize_embeddings=True))
        E_tr, E_te = enc(train_texts), enc(test_texts)
        return E_tr, E_te, name, int(E_tr.shape[1])
    raise RuntimeError(f"no sentence encoder could be loaded: {last}")


def encode_lsa(train_texts, test_texts, dim=300):
    """TF-IDF reduced by truncated SVD -- the fallback allowed by Sec. 1.9 / 3.11."""
    vec = tfidf_vectorizer()
    Xtr, Xte = vec.fit_transform(train_texts), vec.transform(test_texts)
    dim = min(dim, Xtr.shape[1] - 1)
    pipe = make_pipeline(TruncatedSVD(n_components=dim, random_state=SEED),
                         Normalizer(copy=False))
    E_tr, E_te = pipe.fit_transform(Xtr), pipe.transform(Xte)
    ev = float(pipe.named_steps["truncatedsvd"].explained_variance_ratio_.sum())
    return E_tr, E_te, f"TF-IDF + truncated SVD (LSA, {dim}d, {ev:.1%} var)", dim


def build_embeddings(tr, te, pref="auto"):
    """
    Encode once, cache, and reuse.

    Cached at results/cache/emb_<pref>.npz. Delete that file to force a
    re-encode; nothing else needs to change.
    """
    key = os.path.join(CACHE, f"emb_{pref}.npz")
    if os.path.exists(key):
        z = np.load(key, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        print(f"  [cache] embeddings reloaded -- {meta['encoder']} ({meta['dim']}d)")
        return z["E_tr"], z["E_te"], meta["encoder"], meta["dim"], meta["encode_seconds"]

    t0 = time.time()
    if pref == "lsa":
        E_tr, E_te, name, dim = encode_lsa(raw_text(tr), raw_text(te))
    else:
        try:
            E_tr, E_te, name, dim = encode_sbert(raw_text(tr), raw_text(te))
        except Exception as e:
            if pref == "sbert":
                raise
            print(f"  [encoder] falling back to LSA as allowed by Sec. 3.11 ({e})")
            E_tr, E_te, name, dim = encode_lsa(raw_text(tr), raw_text(te))
    secs = round(time.time() - t0, 1)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(key, E_tr=E_tr, E_te=E_te,
                        meta=json.dumps({"encoder": name, "dim": dim,
                                         "encode_seconds": secs}))
    print(f"  encoder: {name}  dim={dim}  encode={secs}s")
    return E_tr, E_te, name, dim, secs
