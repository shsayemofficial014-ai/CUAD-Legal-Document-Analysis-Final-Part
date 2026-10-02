# AI-Powered Legal Document Analysis System (CUAD)

**Clause Detection, Risk Classification and Plain-English Summarization**
B.Sc. Thesis · Department of Computer Science and Engineering · BRAC University

Jerin Aktar (22101279) · Afnan Mazumdar (24141229) · Shoyeb Hasan Sayem (22101386)
Supervisor: Utsho Kumar Roy

---

## What this project is

Commercial contracts run to tens of thousands of characters, and the clauses that matter most are buried
inside them. CUAD (510 expert-annotated contracts, 41 clause categories) framed this as a pure extraction
benchmark: given a category, find the clause. This project takes that further into something a
non-lawyer can act on — it decides **what is in a contract**, **where each clause is**, and then, the
part CUAD does not cover, **what it means and how risky it is**.

Three models, all trained on a single Apple-Silicon laptop:

| Stage | Model | Job | Held-out result |
|---|---|---|---|
| **2A** | TF-IDF + DistilBERT (MIL) ensemble | Which of the 41 categories are present? | micro-F1 **0.776**, accuracy **85.49%** |
| **2B** | DistilBERT-QA | Locate the exact clause text | token-F1 **0.763** |
| **1** | FLAN-T5-small | Rewrite the clause in plain English | ROUGE-L **0.775** |

Every number is measured on a **contract-level 80/20 split** — 408 contracts for training, 102 never-seen
contracts for testing — so no clause from a training contract can appear in the test set.

---

## Start here

| If you want… | Read |
|---|---|
| **The full thesis** | **`thesis/main.pdf`** — 55 pages, 8 chapters, the complete work |
| A step-by-step account of what was done | `PROJECT_REPORT.md` |
| The findings with their caveats | `REPORT.md` |
| *Why* each design choice was made | `METHODOLOGY.md` |
| The code, cell by cell | `notebooks/README.md` |

---

## The three findings

1. **A simple lexical baseline beats a windowed transformer at document-level presence classification.**
   Full-document TF-IDF scores 0.775; the sliding-window DistilBERT scores 0.694. A cluster bootstrap
   confirms the gap is real (+0.081, 95% CI [+0.063, +0.099], significant in 10,000/10,000 resamples).
   This is the result we stand behind most firmly, and it is the opposite of what we expected.

2. **Measure feasibility ceilings before training, not after.** A retrieval hit-rate check that took
   seconds showed that question-guided retrieval capped attainable recall at 64% — below what the cheap
   baseline already achieved. That one measurement killed the retrieval architecture before any training
   time was spent on it.

3. **A high automatic metric can measure the wrong thing.** The summarizer's ROUGE-L of 0.775 looks
   strong, but a zero-shot-vs-fine-tuned comparison shows the model learned to *classify* the clause and
   emit its category template verbatim. With only 41 unique targets, fine-tuning quietly turned
   summarization into classification. We report that rather than let the number overclaim.

**One honest correction we made to ourselves:** the AND-ensemble's 0.776 nominally tops the leaderboard,
but a bootstrap confidence interval shows its margin over the 0.775 baseline is **statistically a tie**
([−0.007, +0.009], leading in only 62.6% of resamples). We keep the ensemble for its balanced error
profile (precision 77.2% / recall 78.0%), not for its rank.

---

## Layout

```
final project p3/
├── README.md                  — this file
├── thesis/                    — the thesis: LaTeX source + compiled PDF
│   ├── main.tex, main.pdf
│   ├── chapters/              — one .tex per chapter
│   └── figures/               — generated plots
├── PROJECT_REPORT.md / .pdf   — complete step-by-step report
├── REPORT.md / .pdf           — findings-focused report
├── METHODOLOGY.md / .pdf      — design-decision record
├── analysis/                  — scripts producing the reported statistics
├── data/                      — the exported 80/20 split (train_80.csv, test_20.csv)
└── notebooks/
    ├── README.md              — cell-by-cell guide
    ├── train.ipynb            — training
    ├── test.ipynb             — evaluation (regenerates every reported number)
    ├── eda.ipynb              — original combined notebook
    ├── artifacts/             — the split + fitted TF-IDF baseline
    └── outputs/               — the three trained model checkpoints
```

---

## Reproducing the results

```bash
pip install -r requirements.txt
```

`notebooks/test.ipynb` loads the saved checkpoints from `notebooks/outputs/` and regenerates every
metric reported here — no re-training needed. The statistical tests are standalone:

```bash
python analysis/bootstrap_ci.py
```

That script independently reproduces the point estimates (0.776 / 0.775 / 0.718 / 0.696 / 0.694) and the
confusion matrix (TN=2523, FP=311, FN=296, TP=1052) before computing the bootstrap intervals, so it
doubles as a check that the saved models still behave as reported.

To rebuild the thesis PDF (requires `tectonic`):

```bash
cd thesis && tectonic main.tex
```

---

## Scope and limitations

- **Backbones are distilled.** DistilBERT and FLAN-T5-small, chosen for laptop reproducibility. CUAD's
  published baselines use DeBERTa-v3-XL, roughly an order of magnitude larger, so our absolute extraction
  scores are not directly comparable to theirs and we do not present them as such.
- **The rare tail is near few-shot.** Five categories have ≤7 positive test contracts and score 0.000.
  That is a property of a 510-contract dataset, not an architecture failure, which is why we quote
  micro- rather than macro-averaged headline numbers.
- **Span extraction is scoped.** Our token-F1 measures extraction *given an answer-bearing window*, not
  CUAD's full-document precision-at-80%-recall benchmark, which we did not run.
- **Gradient-norm traces were not retained.** Checkpoint directories were cleared after training, so
  learning-rate schedules can be reconstructed exactly but per-step gradient norms would need a re-run.
- **This is a research prototype, not legal advice**, and the application says so.
