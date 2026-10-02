"""Train (or reuse) a window-level presence DistilBERT and evaluate it on the
held-out 102 test contracts.

  --tag       run name (output dir + result files)
  --seed      training seed
  --max-train subsample cap; 0 = use ALL 53,662 window examples
  --eval-only path to an existing checkpoint (skips training)

Everything else -- windowing, negative sampling, hyperparameters, the
max-pooling inference rule -- is identical to the thesis run.
"""
import os, sys, json, time, argparse, random
import numpy as np, pandas as pd, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common
from common import NB

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-train", type=int, default=24000)
ap.add_argument("--eval-only", default=None)
A = ap.parse_args()

OUT = os.path.join(HERE, "runs", A.tag)
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[{A.tag}] device={DEVICE} seed={A.seed} max_train={A.max_train or 'ALL'}", flush=True)

random.seed(A.seed); np.random.seed(A.seed); torch.manual_seed(A.seed)
D = common.load_all()

# ------------------------------------------------------------------ training
ckpt = A.eval_only
train_minutes = None
if ckpt is None:
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments)
    from datasets import Dataset

    win_df = common.build_window_examples(D, seed=A.seed)
    if A.max_train and len(win_df) > A.max_train:
        win_df = (win_df.groupby("label", group_keys=False)
                  .apply(lambda g: g.sample(round(A.max_train * len(g) / len(win_df)),
                                            random_state=A.seed)))
    print(f"[{A.tag}] training examples {len(win_df)} pos={win_df.label.mean():.3f}", flush=True)

    MODEL, MAX_LEN, BATCH, EPOCHS, LR = "distilbert-base-uncased", 256, 16, 2, 2e-5
    tok = AutoTokenizer.from_pretrained(MODEL)
    ds = Dataset.from_pandas(win_df[["question", "window", "label"]],
                             preserve_index=False).shuffle(seed=A.seed)
    sp = ds.train_test_split(test_size=0.05, seed=A.seed)

    def enc(b):
        e = tok(b["question"], b["window"], truncation=True, max_length=MAX_LEN)
        e["labels"] = b["label"]
        return e

    tr = sp["train"].map(enc, batched=True, remove_columns=sp["train"].column_names)
    va = sp["test"].map(enc, batched=True, remove_columns=sp["test"].column_names)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    args = TrainingArguments(
        output_dir=f"{OUT}/ckpt", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
        learning_rate=LR, eval_strategy="epoch", logging_strategy="steps",
        logging_steps=100, save_strategy="no", report_to="none", seed=A.seed,
        fp16=False, bf16=False, dataloader_pin_memory=False)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok))
    t0 = time.time()
    trainer.train()
    train_minutes = (time.time() - t0) / 60
    print(f"[{A.tag}] training took {train_minutes:.1f} min", flush=True)
    json.dump(trainer.state.log_history, open(f"{OUT}/log_history.json", "w"), indent=1)
    ckpt = f"{OUT}/final"
    model.save_pretrained(ckpt); tok.save_pretrained(ckpt)

# ------------------------------------------------------------------ inference
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(DEVICE).eval()

te = D["test_titles"]
proba_tfidf, cats = common.tfidf_probs(D, te)
wins_by_title = {t: common.make_windows(D["ctx"][t]) for t in te}
rows = [(ti, cj, D["cat_question"][c], w)
        for ti, t in enumerate(te) for cj, c in enumerate(cats)
        for w in wins_by_title[t]]
print(f"[{A.tag}] scoring {len(rows)} (question, window) pairs", flush=True)

B, probs = 64, np.zeros(len(rows))
t0 = time.time()
for i in range(0, len(rows), B):
    ch = rows[i:i + B]
    e = tok([r[2] for r in ch], [r[3] for r in ch], truncation=True,
            max_length=256, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs[i:i + B] = F.softmax(model(**e).logits, -1)[:, 1].float().cpu().numpy()
    if (i // B) % 400 == 0:
        el = (time.time() - t0) / 60
        print(f"  {i}/{len(rows)}  {el:.1f}m  eta {el/max(i,1)*(len(rows)-i):.1f}m", flush=True)

proba_trans = np.zeros((len(te), len(cats)))
for (ti, cj, _, _), p in zip(rows, probs):
    if p > proba_trans[ti, cj]:
        proba_trans[ti, cj] = p

Y = common.labels(D, te, cats)
pred = {"AND":  (np.minimum(proba_tfidf, proba_trans) >= .5).astype(int),
        "OR":   (np.maximum(proba_tfidf, proba_trans) >= .5).astype(int),
        "AVG":  (((proba_tfidf + proba_trans) / 2) >= .5).astype(int),
        "TRANS": (proba_trans >= .5).astype(int),
        "TFIDF": (proba_tfidf >= .5).astype(int)}
res = {k: common.micro_f1(Y, v) for k, v in pred.items()}
cm = {}
for k, v in pred.items():
    cm[k] = dict(tn=int(((Y == 0) & (v == 0)).sum()), fp=int(((Y == 0) & (v == 1)).sum()),
                 fn=int(((Y == 1) & (v == 0)).sum()), tp=int(((Y == 1) & (v == 1)).sum()))

np.savez_compressed(f"{OUT}/probs.npz", proba_trans=proba_trans,
                    proba_tfidf=proba_tfidf, Y=Y,
                    titles=np.array(te, dtype=object), cats=np.array(cats, dtype=object))
json.dump({"tag": A.tag, "seed": A.seed, "max_train": A.max_train,
           "train_minutes": train_minutes, "micro_f1": res, "confusion": cm},
          open(f"{OUT}/result.json", "w"), indent=2)
print(f"[{A.tag}] " + "  ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
