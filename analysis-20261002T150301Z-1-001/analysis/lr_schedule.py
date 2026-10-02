"""Learning-rate schedules actually used, read off the saved TrainingArguments.

presence_mil / span: lr_scheduler_type=LINEAR, warmup_steps=0, epochs=2 (confirmed
from outputs/*/final/training_args.bin). summarizer: same scheduler family, config
as reported in Table 5.1 (its training_args.bin was not saved with the checkpoint).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.expanduser("~/Desktop/final project p3/thesis/figures")

runs = [
    ("Presence (MIL), DistilBERT", 24000, 16, 2, 2e-5, "#4C78A8"),
    ("Span (QA), DistilBERT",       8000, 16, 2, 3e-5, "#F58518"),
    ("Summarizer, FLAN-T5-small",   4000,  8, 2, 3e-4, "#54A24B"),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
for ax, (name, n, bs, ep, lr, c) in zip(axes, runs):
    total = (n // bs) * ep
    steps = np.arange(total + 1)
    # HF linear scheduler with num_warmup_steps=0: lr * max(0, (T - t) / T)
    sched = lr * np.maximum(0.0, (total - steps) / total)
    ax.plot(steps, sched, color=c, lw=1.8)
    ax.axvline(total / 2, ls="--", lw=0.9, color="#888")
    ax.text(total / 2, lr * 0.97, " epoch 2", fontsize=7.5, color="#666", va="top")
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("optimizer step", fontsize=8.5)
    ax.set_ylabel("learning rate", fontsize=8.5)
    ax.set_xlim(0, total)
    ax.set_ylim(0, lr * 1.08)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=.25, lw=.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.annotate(f"peak {lr:.0e}\n{total:,} steps", xy=(total * 0.55, lr * 0.55),
                fontsize=7.5, color="#555")

plt.tight_layout()
out = f"{FIG}/lr_schedule.png"
plt.savefig(out, dpi=200)
print("wrote", out)
for name, n, bs, ep, lr, _ in runs:
    print(f"  {name}: {(n//bs)*ep:,} steps, peak {lr}")
