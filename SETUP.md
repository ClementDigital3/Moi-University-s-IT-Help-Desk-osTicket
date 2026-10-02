# Running this on another laptop

Start to finish on a machine that has never seen this project.

---

## 1. What you need first

- **Python 3.12** — check with `python3 --version`
- **git**
- **An internet connection for the first run only.** Dependencies and one
  sentence-transformer model (~90 MB) download once; after that everything runs
  offline.
- **About 3 GB free** — most of it is PyTorch.

---

## 2. Clone and install

```bash
git clone https://github.com/ClementDigital3/Moi-University-s-IT-Help-Desk-osTicket.git
cd Moi-University-s-IT-Help-Desk-osTicket

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

That takes a few minutes. Versions are pinned, so you get exactly what the
reported results were produced with.

> **Don't skip `requirements.txt` and install by hand.** It pins the **CPU**
> build of PyTorch. The default PyPI wheel pulls about 3 GB of CUDA packages
> that nothing here uses.

Check it worked:

```bash
make status
```

You'll see a mix of `ok` and `MISSING`, which is correct on a fresh clone. The
screened corpus and the result tables **are** in the repository, so you can read
the findings without running anything. The trained models are **not** — they're
too large and they regenerate — so anything that needs a fitted model says
`MISSING` until the next step.

---

## 3. Build everything

```bash
make all
```

**Expect 45–60 minutes.** Most of it is one step: trying every candidate
classifier for each decision. It prints what it's doing throughout.

This runs, in order: preprocessing → the four models → comparison and figures →
Chapters Four and Five → the dashboard.

`make all` is safe to re-run. Finished work is cached, so a second run takes
seconds. If it's interrupted, just run it again — it resumes rather than
starting over.

It regenerates the committed tables too, so you can confirm you reproduce the
published numbers rather than taking them on trust.

When it finishes, `make status` is `ok` down to `serving bundles`, which the
next step handles.

---

## 4. Run the prototype

```bash
make export     # save the trained models so they can score live tickets (~2 min)
make demo       # starts the server
```

Open **http://127.0.0.1:8000/** and you get six views:

| | |
|---|---|
| `/` | **Arm A** — one model makes all three decisions |
| `/classification` | **Arm B · B1** — category |
| `/routing` | **Arm B · B2** — resolver team |
| `/recommendation` | **Arm B · B3** — prior resolutions |
| `/cascade` | **Arm B** — B1 → B2 → B3 chained |
| `/dashboard` | the findings, with a present mode |

`Ctrl+C` stops the server.

---

## That's it

Those are the only four commands:

```bash
.venv/bin/pip install -r requirements.txt
make all
make export
make demo
```

Run `make` on its own at any time to see everything available.

---

## If something goes wrong

**`make: command not found`** — install it: `sudo apt install make` on
Ubuntu/Debian, `xcode-select --install` on macOS.

**`python3: command not found`, or a version below 3.12** — install Python 3.12
and use it explicitly: `python3.12 -m venv .venv`.

**The install fails on PyTorch** — you're probably not on Python 3.12, or the
machine is 32-bit. Check `.venv/bin/python --version`.

**A model download fails or hangs** — that's the sentence encoder fetching from
huggingface.co on first run. Check the connection and re-run; it resumes. If
the machine can't reach it at all, everything still works on the lexical
representation:

```bash
.venv/bin/python -m models.classification.run --encoder lsa
```

**`make demo` says a bundle is missing** — run `make export` first.

**The dashboard page is blank** — run `make dashboard`.

**You want to start completely clean:**

```bash
make clean-cache      # drop cached models, forces a full refit (slow)
make clean-results    # drop all generated tables and figures
```

---

## A note on the data

`data/raw/` is a **synthetic** corpus, generated to match the structure of the
real osTicket export, pending institutional data-access approval. It is in the
repository so the project runs end to end for anyone who clones it.

The comparative findings hold; the absolute numbers belong to this corpus, not
to Moi University's help desk. To use the real export: replace
`data/raw/osticket_export.csv` (same columns), supply
`data/raw/ground_truth_problem_ids.csv`, and run `make all`. Every table, figure
and chapter regenerates.
