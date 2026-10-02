"""Assemble the training-budget and seed-variance results.

  * full53k vs base24k          -- does training on all 53,662 windows close
                                   the gap to the TF-IDF baseline?
  * seeds 42/43/44 at 24k       -- how big is run-to-run variance compared with
                                   the +0.0014 ensemble margin?

All comparisons use the same paired cluster bootstrap over the 102 test
contracts as Section 6.1.2.
"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

def micro(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)

def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

runs = {}
for f in sorted(glob.glob(f"{HERE}/runs/*/probs.npz")):
    tag = os.path.basename(os.path.dirname(f))
    z = np.load(f, allow_pickle=True)
    runs[tag] = dict(pt=z["proba_trans"], pf=z["proba_tfidf"], Y=z["Y"],
                     cats=list(z["cats"]), titles=list(z["titles"]))
print("runs found:", list(runs))
Y = runs[list(runs)[0]]["Y"]
pf = runs[list(runs)[0]]["pf"]

preds = {}
for tag, r in runs.items():
    preds[f"{tag}:TRANS"] = (r["pt"] >= .5).astype(int)
    preds[f"{tag}:AND"] = (np.minimum(r["pf"], r["pt"]) >= .5).astype(int)
preds["TFIDF"] = (pf >= .5).astype(int)

out = {"point_estimates": {}}
print("\n=== point estimates (micro-F1 / precision / recall) ===")
for k, p in preds.items():
    pr, rc, f1 = prf(Y, p)
    out["point_estimates"][k] = dict(micro_f1=f1, precision=pr, recall=rc)
    print(f"  {k:<26s} F1={f1:.4f}  P={pr:.4f}  R={rc:.4f}")

rng = np.random.default_rng(42); n = len(Y); B = 10000
idxs = [rng.integers(0, n, n) for _ in range(B)]
def contrast(a, b):
    d = np.array([micro(Y[i], preds[a][i]) - micro(Y[i], preds[b][i]) for i in idxs])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=micro(Y, preds[a]) - micro(Y, preds[b]),
                lo=float(lo), hi=float(hi), p_better=float((d > 0).mean()))

pairs = []
if "full53k_seed42:TRANS" in preds:
    pairs += [("full53k_seed42:TRANS", "base24k_seed42:TRANS"),
              ("TFIDF", "full53k_seed42:TRANS"),
              ("full53k_seed42:AND", "TFIDF"),
              ("full53k_seed42:AND", "base24k_seed42:AND")]
pairs += [("TFIDF", "base24k_seed42:TRANS"), ("base24k_seed42:AND", "TFIDF")]
out["contrasts"] = {}
print("\n=== paired cluster bootstrap (10,000 resamples of the 102 contracts) ===")
for a, b in pairs:
    if a not in preds or b not in preds: continue
    c = contrast(a, b); out["contrasts"][f"{a} - {b}"] = c
    print(f"  {a:<26s} - {b:<26s} {c['diff']:+.4f}  95% CI [{c['lo']:+.4f}, {c['hi']:+.4f}]  "
          f"P(>0)={c['p_better']:.3f}  {'TIE' if c['lo'] <= 0 <= c['hi'] else 'DIFFERENT'}")

seeds = [t for t in runs if t.startswith("sub24k") or t == "base24k_seed42"]
if len(seeds) >= 2:
    out["seed_variance"] = {}
    print(f"\n=== seed variance over {len(seeds)} runs at the 24k budget ===")
    for arm in ["TRANS", "AND"]:
        v = np.array([prf(Y, preds[f"{s}:{arm}"])[2] for s in seeds])
        out["seed_variance"][arm] = dict(seeds=seeds, f1=[float(x) for x in v],
                                         mean=float(v.mean()), sd=float(v.std(ddof=1)),
                                         min=float(v.min()), max=float(v.max()),
                                         range=float(v.max() - v.min()))
        print(f"  {arm:<6s} " + "  ".join(f"{s.split('_')[-1]}={x:.4f}" for s, x in zip(seeds, v))
              + f"   mean={v.mean():.4f} sd={v.std(ddof=1):.4f} range={v.max()-v.min():.4f}")
json.dump(out, open(f"{HERE}/compare_runs.json", "w"), indent=2)
print("\nwrote compare_runs.json")
