"""
Reproduces cells 1-5 of notebooks/test.ipynb, then computes a contract-level
bootstrap confidence interval on the AND-ensemble vs TF-IDF micro-F1 difference.

Uses the LOCAL CUADv1.json (identical to the HF copy) so no network is needed.
Caches per-(contract, category) probabilities so re-runs are instant.
"""
import os, json, time, pickle, sys
from collections import defaultdict
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

ROOT = os.path.expanduser("~/Desktop/final project p3/notebooks")
CUAD = os.path.expanduser("~/p3_notebook/CUADv1.json")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(SCRATCH, "probs_cache.npz")
os.chdir(ROOT)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE, flush=True)

split = json.load(open("artifacts/split.json"))
_test = set(split["test"])

js = json.load(open(CUAD))["data"]
cat_from_qid = lambda qid: qid.rsplit("__", 1)[-1]
ctx_by_title = {c["title"]: c["paragraphs"][0]["context"] for c in js}

WIN, STRIDE = 2000, 1500
def make_windows(text):
    wins, pos = [], 0
    while pos < len(text):
        wins.append(text[pos:pos + WIN])
        if pos + WIN >= len(text): break
        pos += STRIDE
    return wins or [text]

gold_spans = defaultdict(list)
for c in js:
    for qa in c["paragraphs"][0]["qas"]:
        for a in qa["answers"]:
            if a["text"].strip():
                gold_spans[(c["title"], cat_from_qid(qa["id"]))].append(a["text"])

bl = pickle.load(open("artifacts/baseline.pkl", "rb"))
vec, classifiers, cats, cat_question = bl["vec"], bl["classifiers"], bl["cats"], bl["cat_question"]

te_titles = [t for t in [c["title"] for c in js] if t in _test]
print("test contracts:", len(te_titles), "categories:", len(cats), flush=True)

# ---- TF-IDF baseline probabilities -----------------------------------------
X_test = vec.transform([ctx_by_title[t] for t in te_titles])
proba_tfidf = np.zeros((len(te_titles), len(cats)), dtype=float)
for j, cat in enumerate(cats):
    kind, obj = classifiers[cat]
    proba_tfidf[:, j] = float(obj) if kind == "const" else obj.predict_proba(X_test)[:, 1]

# ---- Transformer max-pooled probabilities (cached) --------------------------
if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    proba_trans = z["proba_trans"]
    assert list(z["titles"]) == te_titles and list(z["cats"]) == list(cats)
    print("loaded cached transformer probabilities", flush=True)
else:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained("outputs/presence_mil/final")
    model = AutoModelForSequenceClassification.from_pretrained(
        "outputs/presence_mil/final").to(DEVICE).eval()
    MAX_LEN, B = 256, 64

    windows_by_title = {t: make_windows(ctx_by_title[t]) for t in te_titles}
    rows = []
    for ti, t in enumerate(te_titles):
        for cj, cat in enumerate(cats):
            q = cat_question[cat]
            for w in windows_by_title[t]:
                rows.append((ti, cj, q, w))
    print("(question, window) pairs:", len(rows), flush=True)

    probs = np.zeros(len(rows))
    t0 = time.time()
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]
        enc = tok([r[2] for r in chunk], [r[3] for r in chunk], truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        probs[i:i + B] = F.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        if (i // B) % 200 == 0:
            done = i + len(chunk)
            el = time.time() - t0
            eta = el / max(done, 1) * (len(rows) - done) / 60
            print(f"  {done}/{len(rows)}  elapsed {el/60:.1f}m  eta {eta:.1f}m", flush=True)

    proba_trans = np.zeros((len(te_titles), len(cats)), dtype=float)
    acc = defaultdict(float)
    for (ti, cj, _, _), p in zip(rows, probs):
        if p > acc[(ti, cj)]:
            acc[(ti, cj)] = p
    for (ti, cj), p in acc.items():
        proba_trans[ti, cj] = p
    np.savez_compressed(CACHE, proba_trans=proba_trans,
                        titles=np.array(te_titles, dtype=object),
                        cats=np.array(list(cats), dtype=object))
    print(f"inference done in {(time.time()-t0)/60:.1f} min", flush=True)

# ---- Labels and predictions -------------------------------------------------
Y = np.array([[int(bool(gold_spans.get((t, cat), []))) for cat in cats] for t in te_titles])
pred_tfidf = (proba_tfidf >= 0.5).astype(int)
pred_and = (np.minimum(proba_tfidf, proba_trans) >= 0.5).astype(int)
pred_or = (np.maximum(proba_tfidf, proba_trans) >= 0.5).astype(int)
pred_avg = (((proba_tfidf + proba_trans) / 2) >= 0.5).astype(int)
pred_tr = (proba_trans >= 0.5).astype(int)

def micro_f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    if tp == 0: return 0.0
    pr, rc = tp / (tp + fp), tp / (tp + fn)
    return 2 * pr * rc / (pr + rc)

print("\n=== POINT ESTIMATES (should match Table 6.1) ===", flush=True)
for name, p in [("Ensemble AND", pred_and), ("TF-IDF baseline", pred_tfidf),
                ("Ensemble AVG", pred_avg), ("Ensemble OR", pred_or),
                ("Transformer (max-pool)", pred_tr)]:
    print(f"  {name:<24s} micro-F1 = {micro_f1(Y, p):.4f}", flush=True)

tn = int(((Y == 0) & (pred_and == 0)).sum()); fp = int(((Y == 0) & (pred_and == 1)).sum())
fn = int(((Y == 1) & (pred_and == 0)).sum()); tp = int(((Y == 1) & (pred_and == 1)).sum())
print(f"\nAND confusion: TN={tn} FP={fp} FN={fn} TP={tp}", flush=True)

# ---- Contract-level bootstrap ----------------------------------------------
# Decisions within one contract are correlated, so the resampling unit is the
# contract (cluster bootstrap), not the individual (contract, category) cell.
N_BOOT = 10000
rng = np.random.default_rng(SEED)
n = len(te_titles)
d_and, d_bl, d_diff = np.empty(N_BOOT), np.empty(N_BOOT), np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    yb = Y[idx]
    a = micro_f1(yb, pred_and[idx]); c = micro_f1(yb, pred_tfidf[idx])
    d_and[b], d_bl[b], d_diff[b] = a, c, a - c

def ci(x): return np.percentile(x, [2.5, 97.5])

print(f"\n=== CLUSTER BOOTSTRAP ({N_BOOT:,} resamples of the {n} test contracts) ===", flush=True)
lo, hi = ci(d_and);  print(f"  AND-ensemble micro-F1 : {micro_f1(Y,pred_and):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
lo, hi = ci(d_bl);   print(f"  TF-IDF micro-F1       : {micro_f1(Y,pred_tfidf):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
lo, hi = ci(d_diff)
obs = micro_f1(Y, pred_and) - micro_f1(Y, pred_tfidf)
print(f"  DIFFERENCE (AND - TF-IDF): {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  P(AND > TF-IDF) over resamples = {(d_diff > 0).mean():.3f}")
print(f"  CI contains zero: {lo <= 0 <= hi}")

json.dump({
    "n_test_contracts": n, "n_boot": N_BOOT,
    "and_f1": micro_f1(Y, pred_and), "and_ci": list(ci(d_and)),
    "bl_f1": micro_f1(Y, pred_tfidf), "bl_ci": list(ci(d_bl)),
    "diff": obs, "diff_ci": list(ci(d_diff)),
    "p_and_better": float((d_diff > 0).mean()),
    "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
}, open(os.path.join(SCRATCH, "bootstrap_result.json"), "w"), indent=2)
print("\nwrote bootstrap_result.json", flush=True)
