"""Does the AND rule punish rare categories harder than common ones?

Compares the conjunctive ensemble against the transformer alone (and the
TF-IDF baseline) separately on the rare tail and on the common categories,
using the frequency ranking of the TRAINING split.
"""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common

z = np.load(f"{HERE}/runs/base24k_seed42/probs.npz", allow_pickle=True)
pt, pf, Y = z["proba_trans"], z["proba_tfidf"], z["Y"]
cats = list(z["cats"]); titles = list(z["titles"])

D = common.load_all()
tr = set(D["train_titles"])
train_freq = np.array([sum(bool(D["gold_spans"].get((t, c), [])) for t in tr) for c in cats])
test_pos = Y.sum(0)

P = {"AND": (np.minimum(pf, pt) >= .5).astype(int),
     "TRANS": (pt >= .5).astype(int),
     "TFIDF": (pf >= .5).astype(int)}

def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=pr, recall=rc, f1=f1)

order = np.argsort(train_freq)          # ascending: rarest first
rare = order[:10]; common_c = order[10:]

report = {"rare_categories": [{"category": cats[j], "train_pos": int(train_freq[j]),
                               "test_pos": int(test_pos[j])} for j in rare]}
print("=== bottom 10 categories by TRAINING frequency ===")
for j in rare:
    print(f"  {cats[j]:<38s} train_pos={train_freq[j]:3d}  test_pos={test_pos[j]:2d}")

for name, idx in [("rare10", rare), ("common31", common_c), ("all41", np.arange(len(cats)))]:
    report[name] = {}
    print(f"\n=== {name} ({len(idx)} categories) ===")
    for m in ["AND", "TRANS", "TFIDF"]:
        r = prf(Y[:, idx], P[m][:, idx])
        report[name][m] = r
        print(f"  {m:<6s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})")
    a, t = report[name]["AND"], report[name]["TRANS"]
    report[name]["recall_cost_vs_trans"] = t["recall"] - a["recall"]
    report[name]["precision_gain_vs_trans"] = a["precision"] - t["precision"]
    print(f"  AND vs TRANS: recall {a['recall']-t['recall']:+.3f}   "
          f"precision {a['precision']-t['precision']:+.3f}   F1 {a['f1']-t['f1']:+.3f}")

# per-category detail for the rare tail
rows = []
for j in rare:
    r = {"category": cats[j], "train_pos": int(train_freq[j]), "test_pos": int(test_pos[j])}
    for m in ["TRANS", "AND"]:
        s = prf(Y[:, [j]], P[m][:, [j]])
        r[f"{m}_P"] = round(s["precision"], 3); r[f"{m}_R"] = round(s["recall"], 3)
        r[f"{m}_F1"] = round(s["f1"], 3); r[f"{m}_tp"] = s["tp"]
    rows.append(r)
report["per_category_rare"] = rows
print("\n" + pd.DataFrame(rows).to_string(index=False))
json.dump(report, open(f"{HERE}/rare_tail.json", "w"), indent=2)
