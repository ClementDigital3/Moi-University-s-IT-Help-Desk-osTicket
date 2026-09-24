"""
MODEL 3 -- Historical-resolution recommendation, semantic retrieval.

Objective Four / RQ4. Referred to as B3 in Chapter Four.

    new ticket
        -> contextual sentence embedding
        -> ranked nearest resolved tickets, optionally narrowed by the category
           model 1 predicted and optionally blended with a lexical channel
        -> the resolutions that were applied to them

Unlike models 1 and 2 this is retrieval, not classification: it commits to no
single answer, it ranks. That is why it is the most forgiving of the three
decisions -- a resolver shown five candidates can discard the wrong ones at a
glance.

Corpus   training tickets that actually carry a resolution note.
Relevant a retrieved ticket sharing the query's underlying problem.

Four variants are evaluated, and the blend weight is chosen by leave-one-out
retrieval on the TRAINING corpus, never on the held-out queries.

Run:  .venv/bin/python -m models.recommendation.run   (requires model 1 to have run)
Outputs:
    results/artifacts/recommendation.npz
    results/tables/t413_recommendation_variants.csv, _m3_recommendation.csv
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from shared.paths import TABLES, ensure_dirs
from shared.data import load_splits
from shared.encoders import build_embeddings, tfidf_vectorizer
from shared.metrics import rank_metrics, save_table
from shared.artifacts import save_artifact, load_artifact

BLEND_GRID = [0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]


def choose_blend(Ec, C_lex, cpid):
    """
    Pick the semantic/lexical mix by leave-one-out retrieval on the training
    corpus. The held-out queries are never consulted.
    """
    Slo_sem = Ec @ Ec.T
    Slo_lex = cosine_similarity(C_lex, C_lex)
    best_w, best_s = 1.0, -1.0
    for w in BLEND_GRID:
        # Blend first, THEN mask the self-match: masking with -inf beforehand
        # would give 0 * -inf = nan at the endpoint weights.
        blend = w * Slo_sem + (1 - w) * Slo_lex
        np.fill_diagonal(blend, -np.inf)
        hit = (cpid[blend.argmax(1)] == cpid).mean()
        print(f"    w_semantic={w:<5.2f} leave-one-out Top-1 on train = {hit:.4f}")
        if hit > best_s:
            best_w, best_s = w, hit
    print(f"    -> blend weight selected: w_semantic={best_w}")
    return best_w


def evaluate(S, label, model_name, cpid, ccat, qpid, restrict=None, k=10):
    ranked = []
    for i in range(S.shape[0]):
        s = S[i]
        if restrict is not None:
            m = ccat == restrict[i]
            if m.sum() >= k:
                s = np.where(m, s, -np.inf)
        ranked.append([cpid[j] for j in np.argsort(-s)[:k]])
    m = rank_metrics(ranked, qpid)
    m.update({"Task": "Resolution recommendation", "Model": model_name,
              "Variant": label, "Arm": "B (task-specific)"})
    print(f"    {label:34s} Top-1={m['Top-1']:.4f}  Top-5={m['Top-5']:.4f}  "
          f"MRR={m['MRR']:.4f}")
    hits = np.array([1 if r and r[0] == t else 0 for r, t in zip(ranked, qpid)])
    return m, hits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="auto", choices=["auto", "sbert", "lsa"])
    args = ap.parse_args()
    ensure_dirs()
    t_start = time.time()

    tr, te, meta = load_splits()
    print(f"{'='*72}\nMODEL 3 -- Historical-resolution recommendation\n{'='*72}")

    E_tr, E_te, enc_name, dim, enc_s = build_embeddings(tr, te, args.encoder)

    up, up_meta = load_artifact("classification")
    pred_cat = up["test_pred"].astype(str)
    print(f"  upstream: model 1 ({up_meta.get('head','?')}) supplies the category filter")

    keep = tr["res_clean"].fillna("").str.strip().values != ""
    corpus = tr[keep].reset_index(drop=True)
    Ec = E_tr[keep]
    cpid, ccat = corpus["latent_problem_id"].values, corpus["Category"].values
    qpid = te["latent_problem_id"].values
    print(f"  corpus={len(corpus)} resolved tickets   queries={len(te)}")

    S_sem = E_te @ Ec.T                                  # vectors are unit-norm
    vec = tfidf_vectorizer()
    C_lex = vec.fit_transform(corpus["text"])
    S_lex = cosine_similarity(vec.transform(te["text"]), C_lex)

    print("\n  choosing the semantic/lexical blend (leave-one-out on train):")
    w = choose_blend(Ec, C_lex, cpid)
    S_hyb = w * S_sem + (1 - w) * S_lex

    print("\n  variants:")
    variants, hitmap = [], {}
    for label, S, restrict, mname in (
            ("Semantic retrieval", S_sem, None, "Sentence-embedding cosine"),
            ("Semantic + category context", S_sem, pred_cat,
             "Sentence-embedding cosine, category-filtered"),
            (f"Semantic + lexical hybrid (w={w})", S_hyb, None,
             f"Hybrid semantic+lexical (w={w})"),
            ("Hybrid + category context", S_hyb, pred_cat,
             "Hybrid semantic+lexical, category-filtered")):
        m, h = evaluate(S, label, mname, cpid, ccat, qpid, restrict)
        variants.append(m)
        hitmap[label] = h

    save_table(pd.DataFrame(variants)[
        ["Variant", "Top-1", "Top-3", "Top-5", "Top-10", "Precision@5",
         "Recall@5", "MRR"]],
        "t413_recommendation_variants.csv",
        "Table 4.13 -- recommendation variants (Objective 4)")

    best = max(variants, key=lambda r: r["MRR"])
    print(f"  -> best variant: {best['Variant']} (MRR {best['MRR']:.4f})")
    pd.DataFrame([best]).to_csv(os.path.join(TABLES, "_m3_recommendation.csv"), index=False)

    save_artifact("recommendation",
                  meta={"encoder": enc_name, "blend_w_semantic": w,
                        "upstream": "classification",
                        "corpus_size": int(len(corpus)),
                        "best_variant": best["Variant"],
                        "top1": best["Top-1"], "mrr": best["MRR"],
                        "runtime_seconds": round(time.time() - t_start, 1)},
                  hit1=hitmap[best["Variant"]])
    print(f"\ntotal runtime: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
