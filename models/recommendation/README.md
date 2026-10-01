# Model 3 — Historical-resolution recommendation

Ranks previously resolved tickets by similarity so a resolver can reuse what
worked. Thesis calls this **B3**. Needs model 1 for the category filter.

```bash
make m1 && make m3
```

Retrieval, not classification — it ranks rather than committing. That's why it's
the most forgiving of the three: a resolver discards wrong candidates at a
glance, whereas a wrong routing decision costs a reassignment.

Corpus is the 3,345 training tickets carrying a resolution note. The
semantic/lexical blend weight is chosen by leave-one-out on the *training*
corpus — held-out queries are never consulted.

## Current result

| Variant | Top-1 | Top-5 | MRR |
|---|---|---|---|
| Semantic | 0.877 | 0.946 | 0.905 |
| Semantic + category | 0.896 | 0.920 | 0.907 |
| Hybrid (w=0.5) | 0.897 | 0.948 | 0.918 |
| **Hybrid + category** | **0.910** | 0.929 | **0.918** |

Two things matter more than the margins. **MRR exceeds Top-1 everywhere** — when
the best match isn't first it's usually very close, which is what a short
candidate list needs. And the hybrid beats either channel alone: lexical and
semantic similarity are complementary here.

This is the one decision where Arm B beats Arm A outright (0.910 vs 0.875
Top-1, p=0.00011).

## Prototype

Its own page at `/recommendation`, plus the combined `/cascade` view.
Shows the matched tickets with their resolutions, and the category filter applied.

```bash
make export && make demo     # then http://127.0.0.1:8000/recommendation
```

`predict()` takes model 1's predicted `category` to narrow candidates — the
variant the evaluation selected. Matches below similarity 0.50 are flagged weak
and the interface says when the archive has no close precedent.

## To extend

- **Index the resolutions too**, not just problem descriptions
- **Deduplicate the candidate list** — retrieved tickets often share one
  underlying problem; collapsing them shows 5 *distinct* options
- **Human relevance assessment** — Table 3.3 allows it; stronger than the
  latent-problem-id key used now
