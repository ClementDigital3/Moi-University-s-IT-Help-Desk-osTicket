# Design notes

Things that are easy to get wrong here, and why the code is shaped as it is.
Read once; you won't need it again.

## Don't break these

**Out-of-fold context.** Routing is cascaded on classification. During training
it must consume *out-of-fold* predictions — never predictions from a model that
already saw those rows' labels. Breaking this inflates results silently.
See `shared/artifacts.py`.

**Block scaling.** The sentence embedding has unit row-norm, so each of its 384
dimensions carries ~0.05. A raw one-hot metadata block enters at 1.0 and swamps
every distance-based classifier. During development this cost routing 0.28 macro
F1 and looked like a finding about departments — it was an artefact of encoding
width. `unitise()` and `assemble()` in `shared/context.py` fix it; the weight is
cross-validated. Don't bypass them.

**Train/serve parity.** A ticket typed into the prototype goes through
`shared/text.py` — the same construction the training corpus used, verified
identical on all 4,617 rows. Without it the demo scores differently-shaped text
than the model was fitted on.

**The held-out set is scored once.** Model selection and feature ablation use
cross-validation on the training partition only.

## Choices worth knowing

**Two text views.** TF-IDF gets cleaned text (casing and punctuation add
nothing); the sentence encoder gets raw text (they carry signal). That's part of
the representation being tested, not an inconsistency.

**XGBoost uses fixed parameters,** not a grid — one tuned grid cost 27 minutes
per task. A stated compute limitation, not a claim the defaults are optimal.

**Multinomial Naive Bayes is absent from Arm B** by necessity: it needs
non-negative counts and can't take signed embedding dimensions.

**Table and figure numbering.** The CSV filenames (`t410_`, `t44_`…) do *not*
match the thesis numbers. `analysis/write_chapters.py` assigns Chapter Four
numbers by order of appearance and is the single source of truth. Don't renumber
the CSVs.

## Cache files

| File | Delete to |
|---|---|
| `results/cache/emb_auto.npz` | re-encode the corpus (~2 min) |
| `results/cache/heads_*.json` | re-select classifiers (~25 min per task) |
| `results/cache/probas_*.npz` | refit Arm A's branches |
| `results/deploy/*.joblib` | rebuild the serving bundle |

Head selection writes each result as it finishes, so an interrupted run resumes.

## The design system

`shared/ui.css` is the single definition of colour, type and spacing. Every page
links it at `/static/ui.css`; the dashboard inlines the same file at build time,
because it must also work standalone with no server.

**Colour follows the charts, not the other way round.** The data-visualisation
palette was validated against the colourblind-separation and contrast gates in
both modes, so the interface is drawn from those same ramps rather than a second
scheme that merely sits near it. The division is strict:

- `--series-*` and `--seq-*` encode **data** and are never used for chrome
- `--accent` (the dark end of the same blue ramp) and the ink scale carry the
  **interface**, so nothing in the UI can be mistaken for a data series
- `--good` / `--warn` / `--bad` are **status** and always ship with an icon or
  label, never colour alone

**Dark mode is selected, not inverted.** Each dark step was chosen against the
dark surface. Adding a colour means adding both steps.

**Spacing is `--s1`…`--s7`, all multiples of 4**, so everything lands on one
grid. Type is `--t-xs`…`--t-3xl`. Reach for a token before a literal; a literal
in a page is a small divergence that compounds.

Page-specific CSS is allowed but should stay small — currently under 25 lines
each, for genuinely unique things like the cascade's connector arrows.
