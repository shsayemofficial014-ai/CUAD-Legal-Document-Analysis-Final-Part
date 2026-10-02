"""
Rebuild all artifacts the standalone test.ipynb needs, trained on the 80% split:

  artifacts/split.json                 — the 80/20 contract-level split
  artifacts/baseline.pkl               — TF-IDF vectorizer + per-category LogReg
  outputs/presence_mil/final/          — window-level DistilBERT (presence)
  outputs/span/final/                  — DistilBertForQuestionAnswering (span)
  outputs/summarizer/final/            — FLAN-T5-small (summarizer)

Run once (≈80–90 min on Apple MPS). test.ipynb then loads these from disk.
"""
import os, json, time, random, pickle, ast
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE, flush=True)

ART = "artifacts"
os.makedirs(ART, exist_ok=True)
os.makedirs("outputs/presence_mil/final", exist_ok=True)
os.makedirs("outputs/span/final", exist_ok=True)
os.makedirs("outputs/summarizer/final", exist_ok=True)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
js = json.load(open(hf_hub_download("theatticusproject/cuad", repo_type="dataset",
                                    filename="CUAD_v1/CUAD_v1.json")))["data"]

def cat_from_qid(qid):
    return qid.rsplit("__", 1)[-1]

titles = [c["title"] for c in js]
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(titles))
n_test = int(0.20 * len(titles))
test_set = {titles[i] for i in perm[:n_test]}
train_titles = [t for t in titles if t not in test_set]
test_titles = [t for t in titles if t in test_set]
json.dump({"train": train_titles, "test": test_titles}, open(f"{ART}/split.json", "w"))
print(f"split: {len(train_titles)} train / {len(test_titles)} test", flush=True)

ctx_by_title, presence = {}, []
for c in js:
    t = c["title"]; para = c["paragraphs"][0]; ctx_by_title[t] = para["context"]
    for qa in para["qas"]:
        presence.append({"title": t, "category": cat_from_qid(qa["id"]),
                         "question": qa["question"], "label": 1 if qa["answers"] else 0})
pres_df = pd.DataFrame(presence)
cats = sorted(pres_df["category"].unique())
cat_question = pres_df.groupby("category")["question"].first().to_dict()

gold_spans = defaultdict(list)
for c in js:
    t = c["title"]
    for qa in c["paragraphs"][0]["qas"]:
        for a in qa["answers"]:
            if a["text"].strip():
                gold_spans[(t, cat_from_qid(qa["id"]))].append(a["text"])

_train = set(train_titles)

# --------------------------------------------------------------------------- #
# 1) Baseline: TF-IDF + per-category LogReg (fit on TRAIN) -> pickle
# --------------------------------------------------------------------------- #
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tr_titles = [t for t in titles if t in _train]
lab = pres_df.pivot_table(index="title", columns="category", values="label", aggfunc="max")
Y_train = lab.loc[tr_titles, cats].values
vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
X_train = vec.fit_transform([ctx_by_title[t] for t in tr_titles])
classifiers = {}
for j, cat in enumerate(cats):
    ytr = Y_train[:, j]
    if np.unique(ytr).size < 2:
        classifiers[cat] = ("const", int(ytr[0]))
    else:
        classifiers[cat] = ("clf", LogisticRegression(max_iter=1000, class_weight="balanced").fit(X_train, ytr))
pickle.dump({"vec": vec, "classifiers": classifiers, "cats": cats, "cat_question": cat_question},
            open(f"{ART}/baseline.pkl", "wb"))
print("[1/4] baseline saved", flush=True)

# --------------------------------------------------------------------------- #
# 2) Presence MIL: window-level DistilBERT -> save
# --------------------------------------------------------------------------- #
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
                          DataCollatorWithPadding, DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments)
from datasets import Dataset

WIN, STRIDE = 2000, 1500
def make_windows(text):
    wins, pos = [], 0
    while pos < len(text):
        wins.append(text[pos:pos + WIN])
        if pos + WIN >= len(text): break
        pos += STRIDE
    return wins or [text]
windows_by_title = {t: make_windows(x) for t, x in ctx_by_title.items()}

def has_span(spans, w): return any(s[:80] in w for s in spans)
_r = random.Random(SEED); NEG = 2; win_examples = []
for c in js:
    title = c["title"]
    if title not in _train: continue
    wins = windows_by_title[title]
    for qa in c["paragraphs"][0]["qas"]:
        cat = cat_from_qid(qa["id"]); q = qa["question"]; spans = gold_spans.get((title, cat), [])
        if spans:
            pos_idx = [i for i, w in enumerate(wins) if has_span(spans, w)]
            neg_idx = [i for i in range(len(wins)) if i not in pos_idx]
            for i in pos_idx: win_examples.append({"question": q, "window": wins[i], "label": 1})
            for i in _r.sample(neg_idx, min(NEG, len(neg_idx))): win_examples.append({"question": q, "window": wins[i], "label": 0})
        else:
            for i in _r.sample(range(len(wins)), min(NEG, len(wins))): win_examples.append({"question": q, "window": wins[i], "label": 0})
win_df = pd.DataFrame(win_examples)

MODEL_NAME = "distilbert-base-uncased"; MAX_LEN, BATCH, EPOCHS, LR = 256, 16, 2, 2e-5; MAX_TRAIN = 24000
if len(win_df) > MAX_TRAIN:
    win_df_s = win_df.groupby("label", group_keys=False).apply(
        lambda g: g.sample(round(MAX_TRAIN * len(g) / len(win_df)), random_state=SEED))
else:
    win_df_s = win_df
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
wds = Dataset.from_pandas(win_df_s[["question", "window", "label"]], preserve_index=False).shuffle(seed=SEED)
wsplit = wds.train_test_split(test_size=0.05, seed=SEED)
def enc(b):
    e = tok(b["question"], b["window"], truncation=True, max_length=MAX_LEN); e["labels"] = b["label"]; return e
wtr = wsplit["train"].map(enc, batched=True, remove_columns=wsplit["train"].column_names)
wva = wsplit["test"].map(enc, batched=True, remove_columns=wsplit["test"].column_names)
mil = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
args = TrainingArguments(output_dir="outputs/presence_mil/ckpt", num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH, learning_rate=LR,
    eval_strategy="epoch", logging_strategy="steps", logging_steps=100, save_strategy="no",
    report_to="none", seed=SEED, fp16=False, bf16=False, dataloader_pin_memory=False)
tr = Trainer(model=mil, args=args, train_dataset=wtr, eval_dataset=wva,
             processing_class=tok, data_collator=DataCollatorWithPadding(tok))
t0 = time.time(); tr.train()
mil.save_pretrained("outputs/presence_mil/final"); tok.save_pretrained("outputs/presence_mil/final")
print(f"[2/4] presence model saved ({(time.time()-t0)/60:.1f} min)", flush=True)
del mil, tr
if DEVICE == "mps": torch.mps.empty_cache()

# --------------------------------------------------------------------------- #
# 3) Span: DistilBertForQuestionAnswering -> save
# --------------------------------------------------------------------------- #
span_examples = []
for c in js:
    title = c["title"]
    if title not in _train: continue
    wins = windows_by_title[title]
    for qa in c["paragraphs"][0]["qas"]:
        if not qa["answers"]: continue
        cat, q = cat_from_qid(qa["id"]), qa["question"]
        for a in qa["answers"]:
            at = a["text"].strip()
            if not at: continue
            probe = at[:200]
            for w in wins:
                pos = w.find(probe)
                if pos != -1:
                    span_examples.append({"category": cat, "question": q, "context": w,
                                          "answer_text": at, "char_start": pos}); break
span_df = pd.DataFrame(span_examples)

QA_MODEL, QA_MAXLEN, FOCUS, MAX_SPAN_TRAIN = "distilbert-base-uncased", 320, 1200, 8000
qa_tok = AutoTokenizer.from_pretrained(QA_MODEL)
def encode_qa(batch):
    ctxs, a_starts, a_ends = [], [], []
    for w, cs, at in zip(batch["context"], batch["char_start"], batch["answer_text"]):
        start_f = max(0, cs - 150); ctx = w[start_f:start_f + FOCUS]; a_s = cs - start_f
        ctxs.append(ctx); a_starts.append(a_s); a_ends.append(min(len(ctx), a_s + len(at)))
    enc = qa_tok(batch["question"], ctxs, truncation="only_second", max_length=QA_MAXLEN, return_offsets_mapping=True)
    starts, ends = [], []
    for i in range(len(ctxs)):
        off, seq = enc["offset_mapping"][i], enc.sequence_ids(i)
        idx = 0
        while seq[idx] != 1: idx += 1
        c_start = idx
        while idx < len(seq) and seq[idx] == 1: idx += 1
        c_end = idx - 1
        a_s, a_e = a_starts[i], a_ends[i]
        if off[c_start][0] > a_s or off[c_end][1] < a_e:
            ts = te = 0
        else:
            j = c_start
            while j <= c_end and off[j][0] <= a_s: j += 1
            ts = j - 1
            j = c_end
            while j >= c_start and off[j][1] >= a_e: j -= 1
            te = j + 1
        starts.append(ts); ends.append(te)
    enc["start_positions"], enc["end_positions"] = starts, ends
    enc.pop("offset_mapping"); return enc

sp_train = span_df.sample(min(MAX_SPAN_TRAIN, len(span_df)), random_state=SEED)
qa_train = Dataset.from_pandas(sp_train, preserve_index=False).map(
    encode_qa, batched=True, remove_columns=sp_train.columns.tolist())
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
def qa_collator(features):
    batch = qa_tok.pad({"input_ids": [f["input_ids"] for f in features],
                        "attention_mask": [f["attention_mask"] for f in features]},
                       padding=True, return_tensors="pt")
    batch["start_positions"] = torch.tensor([f["start_positions"] for f in features])
    batch["end_positions"] = torch.tensor([f["end_positions"] for f in features])
    return batch
qa_args = TrainingArguments(output_dir="outputs/span/ckpt", num_train_epochs=2,
    per_device_train_batch_size=16, per_device_eval_batch_size=16, learning_rate=3e-5,
    logging_strategy="steps", logging_steps=100, eval_strategy="no", save_strategy="no",
    report_to="none", seed=SEED, fp16=False, bf16=False, dataloader_pin_memory=False)
qa_tr = Trainer(model=qa_model, args=qa_args, train_dataset=qa_train,
                processing_class=qa_tok, data_collator=qa_collator)
t0 = time.time(); qa_tr.train()
qa_model.save_pretrained("outputs/span/final"); qa_tok.save_pretrained("outputs/span/final")
print(f"[3/4] span model saved ({(time.time()-t0)/60:.1f} min)", flush=True)
del qa_model, qa_tr
if DEVICE == "mps": torch.mps.empty_cache()

# --------------------------------------------------------------------------- #
# 4) Summarizer: FLAN-T5-small (CPU) -> save
# --------------------------------------------------------------------------- #
GOLD_SUMMARIES = {
    "Document Name": "This states the official name and type of the agreement.",
    "Parties": "This identifies who is signing the contract and the role each side plays.",
    "Agreement Date": "This is the date the contract was signed.",
    "Effective Date": "This is the date the contract terms actually start applying.",
    "Expiration Date": "This is when the contract is scheduled to end.",
    "Renewal Term": "This explains how the contract continues if neither side cancels.",
    "Notice Period To Terminate Renewal": "This is how long ahead you must say you will not renew.",
    "Governing Law": "This says which country or state law decides any dispute.",
    "Most Favored Nation": "This promises you the best deal the other side gives anyone else.",
    "Non-Compete": "This restricts you from working with the other side's competitors.",
    "Exclusivity": "This makes one side the only allowed source or buyer for something.",
    "No-Solicit Of Customers": "You agree not to approach the other side's customers.",
    "Competitive Restriction Exception": "This lists where the non-compete or exclusivity does not apply.",
    "No-Solicit Of Employees": "You agree not to recruit the other side's employees.",
    "Non-Disparagement": "You agree not to publicly criticise the other side.",
    "Termination For Convenience": "Either side can end the contract for any reason with notice.",
    "Rofr/Rofo/Rofn": "You get the first chance to match an offer or take an opportunity.",
    "Change Of Control": "This explains what happens if one company is bought or merges.",
    "Anti-Assignment": "You cannot transfer this contract to someone else without permission.",
    "Revenue/Profit Sharing": "You share a portion of revenue or profit with the other side.",
    "Price Restrictions": "This limits what prices you can charge or change.",
    "Minimum Commitment": "You promise to buy or deliver at least a stated minimum amount.",
    "Volume Restriction": "There is a cap on how much you can sell or distribute.",
    "Ip Ownership Assignment": "Any new intellectual property created belongs to the named owner.",
    "Joint Ip Ownership": "Both sides jointly own intellectual property created under this contract.",
    "License Grant": "This grants permission to use the other side's intellectual property.",
    "Non-Transferable License": "You cannot pass this licence to anyone else.",
    "Affiliate License-Licensor": "The licensor's affiliates also grant you rights here.",
    "Affiliate License-Licensee": "Your affiliates can use the licence too.",
    "Unlimited/All-You-Can-Eat-License": "The licence is unlimited in scope or volume.",
    "Irrevocable Or Perpetual License": "The licence cannot be taken away and may last forever.",
    "Source Code Escrow": "Source code is held by a third party in case something goes wrong.",
    "Post-Termination Services": "Some services continue for a while after the contract ends.",
    "Audit Rights": "The other side can inspect your records to check compliance.",
    "Uncapped Liability": "There is no limit on how much one side may owe in damages.",
    "Cap On Liability": "There is a maximum amount one side may owe in damages.",
    "Liquidated Damages": "A fixed amount is owed for specific breaches, agreed in advance.",
    "Warranty Duration": "This is how long the warranty lasts.",
    "Insurance": "You must keep specified insurance coverage in force.",
    "Covenant Not To Sue": "You agree not to sue the other side over the listed matters.",
    "Third Party Beneficiary": "Someone not signing the contract still gets to enforce part of it.",
}
rows = []
for c in js:
    if c["title"] not in _train: continue
    for qa in c["paragraphs"][0]["qas"]:
        s = GOLD_SUMMARIES.get(cat_from_qid(qa["id"]))
        if not s: continue
        for a in qa["answers"]:
            at = a["text"].strip()
            if at:
                rows.append({"input_text": f"summarize in plain English: {at[:1200]}", "target_text": s})
sum_df = pd.DataFrame(rows)

T5_MODEL, MAX_IN, MAX_OUT, MAX_SUM_TRAIN = "google/flan-t5-small", 256, 48, 4000
t5_tok = AutoTokenizer.from_pretrained(T5_MODEL)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL)
s_train = sum_df.sample(min(MAX_SUM_TRAIN, len(sum_df)), random_state=SEED)
def enc_sum(b):
    mi = t5_tok(b["input_text"], truncation=True, max_length=MAX_IN)
    mi["labels"] = t5_tok(text_target=b["target_text"], truncation=True, max_length=MAX_OUT)["input_ids"]
    return mi
t5_train = Dataset.from_pandas(s_train, preserve_index=False).map(
    enc_sum, batched=True, remove_columns=s_train.columns.tolist())
t5_args = Seq2SeqTrainingArguments(output_dir="outputs/summarizer/ckpt", num_train_epochs=2,
    per_device_train_batch_size=8, learning_rate=3e-4, logging_strategy="steps", logging_steps=100,
    save_strategy="no", report_to="none", seed=SEED, use_cpu=True, fp16=False, bf16=False,
    dataloader_pin_memory=False)
t5_tr = Seq2SeqTrainer(model=t5_model, args=t5_args, train_dataset=t5_train,
                       processing_class=t5_tok, data_collator=DataCollatorForSeq2Seq(t5_tok, model=t5_model))
t0 = time.time(); t5_tr.train()
t5_model.save_pretrained("outputs/summarizer/final"); t5_tok.save_pretrained("outputs/summarizer/final")
print(f"[4/4] summarizer model saved ({(time.time()-t0)/60:.1f} min)", flush=True)

print("ALL ARTIFACTS BUILT", flush=True)
