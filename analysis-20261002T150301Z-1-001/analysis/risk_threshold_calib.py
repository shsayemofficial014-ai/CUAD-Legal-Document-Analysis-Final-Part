"""Three evaluations the second review asked for, all from saved test-set
probabilities (no retraining):

  1. risk-weighted evaluation  -- per risk band, and the missed-High-risk rate
  2. threshold sweep           -- justifies (or does not justify) the 0.5 default
  3. calibration               -- does the confidence the app shows mean anything?
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
FIG = os.path.expanduser("~/Desktop/final project p3/thesis/figures")

z = np.load(f"{HERE}/runs/base24k_seed42/probs.npz", allow_pickle=True)
pt, pf, Y = z["proba_trans"], z["proba_tfidf"], z["Y"]
cats = list(z["cats"])
risk = json.load(open(f"{HERE}/risk_levels.json"))
band = np.array([risk[c] for c in cats])

def prf(y, p):
    tp = int(((y==1)&(p==1)).sum()); fp = int(((y==0)&(p==1)).sum()); fn = int(((y==1)&(p==0)).sum())
    pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=pr, recall=rc,
                f1=2*pr*rc/(pr+rc) if pr+rc else 0.0)

out = {}
# ---------------------------------------------------------------- 1. by band
P = {"AND": np.minimum(pf,pt)>=.5, "TRANS": pt>=.5, "TFIDF": pf>=.5, "OR": np.maximum(pf,pt)>=.5}
print("=== risk-weighted evaluation (threshold 0.5) ===")
out["by_band"] = {}
for b in ["High","Medium","Low"]:
    idx = np.where(band==b)[0]
    out["by_band"][b] = {"n_categories": len(idx), "n_test_positives": int(Y[:,idx].sum())}
    print(f"\n{b}-risk ({len(idx)} categories, {int(Y[:,idx].sum())} test positives)")
    for m,pred in P.items():
        r = prf(Y[:,idx], pred[:,idx].astype(int)); out["by_band"][b][m] = r
        print(f"   {m:6s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}  missed={r['fn']}")

hi = np.where(band=="High")[0]
for m,pred in P.items():
    r = prf(Y[:,hi], pred[:,hi].astype(int))
    out.setdefault("high_risk_miss_rate", {})[m] = 1-r["recall"]
print("\nmissed High-risk clause rate:", {k: round(v,3) for k,v in out["high_risk_miss_rate"].items()})

# ------------------------------------------------------------ 2. threshold sweep
ths = np.round(np.arange(0.05, 0.96, 0.05), 2)
sweep = {"threshold": ths.tolist()}
for name, score in [("TRANS", pt), ("AND", np.minimum(pf, pt))]:
    rows = []
    for t in ths:
        pred = (score>=t).astype(int)
        a = prf(Y, pred); h = prf(Y[:,hi], pred[:,hi])
        rows.append(dict(t=float(t), precision=a["precision"], recall=a["recall"], f1=a["f1"],
                         high_recall=h["recall"], high_precision=h["precision"]))
    sweep[name] = rows
    best = max(rows, key=lambda r: r["f1"])
    sweep[name+"_best_f1_threshold"] = best["t"]
    print(f"\n{name}: best micro-F1 {best['f1']:.4f} at threshold {best['t']} "
          f"(default 0.5 gives {[r for r in rows if r['t']==0.5][0]['f1']:.4f})")
out["threshold_sweep"] = sweep

# ------------------------------------------------------------ 3. calibration
def calib(score, label, nbins=10):
    edges = np.linspace(0,1,nbins+1); s=score.ravel(); y=label.ravel()
    ece=0.0; rows=[]
    for i in range(nbins):
        m = (s>=edges[i]) & (s<edges[i+1] if i<nbins-1 else s<=1.0)
        if m.sum()==0:
            rows.append(dict(bin=[float(edges[i]),float(edges[i+1])], n=0, conf=None, acc=None)); continue
        conf=float(s[m].mean()); acc=float(y[m].mean()); n=int(m.sum())
        ece += n/len(s)*abs(acc-conf)
        rows.append(dict(bin=[float(edges[i]),float(edges[i+1])], n=n, conf=conf, acc=acc))
    brier = float(((s-y)**2).mean())
    return dict(ece=float(ece), brier=brier, bins=rows)

print("\n=== calibration of the score the app displays as 'confidence' ===")
for name, score in [("TRANS (max-pooled)", pt), ("TFIDF", pf)]:
    c = calib(score, Y); out.setdefault("calibration", {})[name] = c
    print(f"  {name:20s} ECE={c['ece']:.3f}  Brier={c['brier']:.3f}")
    for r in c["bins"]:
        if r["n"]:
            print(f"      [{r['bin'][0]:.1f},{r['bin'][1]:.1f})  n={r['n']:5d}  mean conf={r['conf']:.3f}  actual={r['acc']:.3f}")

json.dump(out, open(f"{HERE}/risk_threshold_calib.json","w"), indent=2)

# ------------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11,4.2))
for name, style in [("TRANS","-o"), ("AND","-s")]:
    r = sweep[name]
    ax[0].plot([x["t"] for x in r], [x["f1"] for x in r], style, ms=3, label=f"{name} micro-F1")
    ax[0].plot([x["t"] for x in r], [x["high_recall"] for x in r], style, ms=3, ls="--", alpha=.6,
               label=f"{name} High-risk recall")
ax[0].axvline(0.5, color="k", lw=.8, ls=":"); ax[0].set_xlabel("confidence threshold")
ax[0].set_ylabel("score"); ax[0].legend(fontsize=7); ax[0].set_title("Threshold sweep")
ax[0].grid(alpha=.3)
for name, score in [("Transformer (max-pool)", pt), ("TF-IDF", pf)]:
    c = out["calibration"]["TRANS (max-pooled)" if "Transf" in name else "TFIDF"]
    xs=[r["conf"] for r in c["bins"] if r["n"]]; ys=[r["acc"] for r in c["bins"] if r["n"]]
    ax[1].plot(xs, ys, "-o", ms=4, label=f"{name} (ECE={c['ece']:.2f})")
ax[1].plot([0,1],[0,1],"k:",lw=.8,label="perfect calibration")
ax[1].set_xlabel("predicted confidence"); ax[1].set_ylabel("observed frequency")
ax[1].set_title("Reliability diagram"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{FIG}/threshold_calibration.png", dpi=160)
print("\nwrote figures/threshold_calibration.png")
