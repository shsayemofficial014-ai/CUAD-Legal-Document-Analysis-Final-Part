"""Additional paired bootstrap contrasts, reusing the cached window probabilities."""
import os, json, pickle
from collections import defaultdict
import numpy as np

ROOT = os.path.expanduser("~/Desktop/final project p3/notebooks")
CUAD = os.path.expanduser("~/p3_notebook/CUADv1.json")
S = "/private/tmp/claude-501/-Users-afnanmazumder-content101/175f0054-ef6b-4873-85c9-ab661952781e/scratchpad"
os.chdir(ROOT)

z = np.load(f"{S}/probs_cache.npz", allow_pickle=True)
proba_trans = z["proba_trans"]; te_titles = list(z["titles"]); cats = list(z["cats"])

js = json.load(open(CUAD))["data"]
ctx = {c["title"]: c["paragraphs"][0]["context"] for c in js}
gold = defaultdict(list)
for c in js:
    for qa in c["paragraphs"][0]["qas"]:
        for a in qa["answers"]:
            if a["text"].strip():
                gold[(c["title"], qa["id"].rsplit("__", 1)[-1])].append(a["text"])

bl = pickle.load(open("artifacts/baseline.pkl", "rb"))
vec, clf = bl["vec"], bl["classifiers"]
X = vec.transform([ctx[t] for t in te_titles])
proba_tfidf = np.zeros_like(proba_trans)
for j, cat in enumerate(cats):
    kind, obj = clf[cat]
    proba_tfidf[:, j] = float(obj) if kind == "const" else obj.predict_proba(X)[:, 1]

Y = np.array([[int(bool(gold.get((t, c), []))) for c in cats] for t in te_titles])
P = {
    "AND":  (np.minimum(proba_tfidf, proba_trans) >= 0.5).astype(int),
    "TFIDF": (proba_tfidf >= 0.5).astype(int),
    "TRANS": (proba_trans >= 0.5).astype(int),
}

def f1(y, p):
    tp = ((y == 1) & (p == 1)).sum(); fp = ((y == 0) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

rng = np.random.default_rng(42); n = len(te_titles); B = 10000
idxs = [rng.integers(0, n, n) for _ in range(B)]

def contrast(a, b):
    d = np.array([f1(Y[i], P[a][i]) - f1(Y[i], P[b][i]) for i in idxs])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return f1(Y, P[a]) - f1(Y, P[b]), lo, hi, (d > 0).mean()

out = {}
for a, b in [("AND", "TFIDF"), ("TFIDF", "TRANS"), ("AND", "TRANS")]:
    obs, lo, hi, pw = contrast(a, b)
    out[f"{a}-{b}"] = dict(diff=obs, lo=lo, hi=hi, p_better=pw)
    print(f"{a:6s} - {b:6s}: {obs:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(>0)={pw:.3f}  {'CONTAINS ZERO' if lo <= 0 <= hi else 'EXCLUDES ZERO'}")

json.dump(out, open(f"{S}/extra_ci.json", "w"), indent=2)
