# Model 3 — Historical-resolution recommendation

**Objective Four / RQ4.** Chapter Four calls this model **B3**.

Given a new ticket, ranks previously resolved tickets by similarity so the
resolver can reuse what worked before.

## Architecture

```
new ticket ──► contextual sentence embedding ─┐
                                              ├─► blended similarity ──► ranked
           ──► TF-IDF lexical vector ─────────┘        │                 resolutions
                                                       │
           predicted category from model 1 ────────────┘  (optional filter)
```

This is **retrieval, not classification**. It commits to no single answer — it
ranks. That is why it is the most forgiving of the three decisions: a resolver
shown five candidates discards the wrong ones at a glance, whereas a wrong
routing decision costs a reassignment.

- **Corpus** — training tickets that actually carry a resolution note (3,345).
- **Relevant** — a retrieved ticket sharing the query's underlying problem.
- **Blend weight** — chosen by leave-one-out retrieval on the *training* corpus.
  The held-out queries are never consulted.

## Run

```bash
.venv/bin/python -m models.classification.run    # required upstream (category filter)
.venv/bin/python -m models.recommendation.run
```

~2 s warm. The embeddings are shared with the other models.

## Current result

| Variant | Top-1 | Top-5 | MRR |
|---|---|---|---|
| Semantic retrieval | 0.8766 | 0.9459 | 0.9051 |
| Semantic + category context | 0.8961 | 0.9199 | 0.9066 |
| Semantic + lexical hybrid (w=0.5) | 0.8972 | 0.9481 | 0.9175 |
| **Hybrid + category context** | **0.9102** | 0.9286 | **0.9183** |

Two things matter more than the margins. **MRR sits above Top-1 in every
variant** — when the best match is not first it is usually very close to first,
which is exactly what a short candidate list needs. And the hybrid beats either
channel alone: lexical and semantic similarity are complementary here, so the
blend is not a hedge.

This is the one decision where the task-specific model beats the end-to-end
model outright (Top-1 0.910 vs 0.875, McNemar p=0.00011).

## Where to extend

- **Resolution text, not just the ticket** — retrieval currently matches
  problem descriptions. Indexing the resolution notes as well, or a
  query-to-resolution bi-encoder, is untried.
- **Deduplicating the candidate list** — several retrieved tickets often share
  one underlying problem; collapsing them would show the resolver five
  *distinct* options instead of five near-copies.
- **Human relevance assessment** — Table 3.3 allows for it, and the current
  relevance key is the latent problem id. Real resolver judgements would be
  stronger evidence.

## Outputs

- `results/artifacts/recommendation.{npz,json}`
- `results/tables/t413_recommendation_variants.csv`, `_m3_recommendation.csv`
