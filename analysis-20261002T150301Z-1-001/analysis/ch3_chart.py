"""Train-vs-test per-category positive-rate chart (Chapter 3) + window-count stats."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/Desktop/final project p3")
FIG = os.path.join(ROOT, "thesis/figures")

tr = pd.read_csv(f"{ROOT}/data/train_80.csv")
te = pd.read_csv(f"{ROOT}/data/test_20.csv")

rt = tr.groupby("category")["present"].mean() * 100
rs = te.groupby("category")["present"].mean() * 100
d = pd.DataFrame({"train": rt, "test": rs}).sort_values("train", ascending=False)

print(f"categories: {len(d)}")
print(f"overall train {tr['present'].mean()*100:.1f}%  test {te['present'].mean()*100:.1f}%")
print(f"max |train-test| gap: {(d['train']-d['test']).abs().max():.1f} pp "
      f"({(d['train']-d['test']).abs().idxmax()})")
print(f"mean |gap|: {(d['train']-d['test']).abs().mean():.1f} pp")
print(f"corr(train, test) = {d['train'].corr(d['test']):.3f}")

# ---- window-count statistics (make_windows: WIN=2000, STRIDE=1500) ----------
def nwin(L):
    if L <= 2000: return 1
    return int((L - 2000) // 1500) + 2
lens = pd.concat([tr, te]).drop_duplicates("contract_title")["context_chars"]
w = lens.apply(nwin)
print(f"\ncontracts: {len(lens)}  median chars {lens.median():.0f}  max {lens.max():.0f}")
print(f"windows/contract: median {w.median():.0f}  mean {w.mean():.1f}  max {w.max():.0f}")

# ---- figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.2))
y = np.arange(len(d))
h = 0.4
ax.barh(y + h/2, d["train"], h, label="Train (408 contracts)", color="#4C78A8")
ax.barh(y - h/2, d["test"],  h, label="Test (102 contracts)",  color="#F58518")
ax.set_yticks(y)
ax.set_yticklabels(d.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Contracts containing the category (%)")
ax.axvline(tr["present"].mean()*100, ls="--", lw=1, color="#4C78A8", alpha=.7)
ax.axvline(te["present"].mean()*100, ls=":",  lw=1.2, color="#F58518", alpha=.9)
ax.text(tr["present"].mean()*100 + 1, len(d)-1.2,
        f"overall: {tr['present'].mean()*100:.1f}% train / {te['present'].mean()*100:.1f}% test",
        fontsize=8, color="#333")
ax.legend(loc="lower right", fontsize=9, frameon=False)
ax.grid(axis="x", alpha=.25, lw=.6)
ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
plt.tight_layout()
out = f"{FIG}/split_distribution.png"
plt.savefig(out, dpi=200)
print("wrote", out)
