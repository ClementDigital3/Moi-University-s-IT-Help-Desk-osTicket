# Model 2 — Resolver routing

Assigns a ticket to one of 7 teams. Thesis calls this **B2**.
Cascaded on model 1 — needs it to have run.

```bash
make m1 && make m2
```

## The cascade is the point

Routing is conditioned on *what kind of problem* the ticket is, not just its
words. It's the most valuable context available:

| Feature set | CV macro F1 | Δ |
|---|---|---|
| Semantic text only | 0.7006 | — |
| + department | 0.7031 | +0.0025 |
| + time | 0.6999 | −0.0032 |
| + text-shape | 0.7025 | +0.0026 |
| **+ predicted category** | **0.7098** | **+0.0073** |

It consumes model 1's **out-of-fold** probabilities during training, never the
true category. Breaking that inflates results silently — see NOTES.md.

## Current result

Linear SVM (C=1.0) · accuracy **0.7294** · macro F1 **0.6888** · correct-routing
rate **0.7294**

Routing is the hardest decision and the costliest to get wrong — a misroute
burns the wrong team's time. Error concentrates on teams described in
overlapping language and on Tier 1, which legitimately receives everything.
Chapter Five therefore recommends deploying it as a *ranked suggestion*.

## Prototype

```bash
make export && make demo     # then http://127.0.0.1:8000/cascade
```

`predict()` requires model 1's `proba_vector` as `category` — it raises rather
than silently guessing, because the cascade is the architecture.

Linear SVM has no probabilities, so scores are **softmaxed margins**, labelled
as such in the interface. That's deliberate: a margin dressed up as a
probability would show confidence the model never claimed, and the thesis
recommends routing be a ranked suggestion anyway.

## To extend

- **Top-k routing** — return 2–3 teams with confidence instead of an argmax.
  This is what the thesis actually recommends and it isn't built yet.
- **Joint model** — the cascade inherits model 1's error; optimising both
  together is the natural comparison
- **Reassignment cost** — the corpus has a `Reassigned` field nothing uses
