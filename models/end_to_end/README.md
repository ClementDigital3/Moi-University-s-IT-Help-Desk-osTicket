# The end-to-end model

One model, one representation, every decision — the baseline Arm B is measured
against. Thesis calls this **Arm A**.

```bash
make end2end    # evaluate
make demo       # live prototype at http://127.0.0.1:8000/
```

Shared TF-IDF fitted once → five Table 3.2 classifiers, each fitted once with
probabilities cached → combined by vote. Plus TF-IDF cosine retrieval.

## Current result

| Task | Best branch | Combined | Gain |
|---|---|---|---|
| Classification | Linear SVM 0.899 | **0.901** (soft vote) | +0.002 |
| Routing | Multinomial NB 0.705 | 0.696 (majority vote) | **−0.009** |

Recommendation: Top-1 0.875, MRR 0.904.

Note row two: **combining was worse than the best single branch.** Voting isn't
free, and the thesis reports that rather than quietly picking the best number.

## The prototype

Python standard library only — no framework, no CDN, no network. Runs with the
wifi off.

- `export.py` fits and saves the bundle (~90s, training partition only)
- `predict.py` inference — importable, testable without a server
- `serve.py` the local server, also serves the dashboard at `/dashboard`
- `app.html` the interface

### What to demo, in order

**"Ambiguous login"** first — the most useful 30 seconds you have. Confidence
collapses to 34%, branches split 3–2, routing still correctly goes to Help Desk
(Tier 1), and every retrieved ticket is titled "Cannot login" yet sits in a
different category. That's Section 1.2's problem statement on screen.

**"Wi-Fi in a lecture hall"** second, for contrast: 100% agreement, full
confidence.

Showing the five individual votes is the point — Table 4.16 lists vote margin as
this model's only way of explaining itself, and the panel gets to watch it
collapse.

### Honesty features

- **Retrieval confidence** — top-1 similarity on held-out data runs 0.44–0.76.
  Above 0.50 retrieval is right 91% of the time; below, 63%. Weak matches are
  coloured differently and the page says when nothing is a close precedent.
- **Train/serve parity** — verified identical on all 4,617 rows (NOTES.md)
- **Held-out set untouched** — everything the demo knows came from training data

### Layout

Fixed frame: compact header, two independently scrolling columns. Verified at
1920×1080, 1366×768, and the short/narrow fallbacks. `/?subject=…&description=…`
prefills and scores server-side for prepared examples.
