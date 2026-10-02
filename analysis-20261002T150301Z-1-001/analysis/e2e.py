"""End-to-end pipeline evaluation.

For every gold (contract, category) pair in the 102 held-out contracts, run the
deployed pipeline start to finish and ask whether the clause survives ALL
stages:

  stage 1  presence detected            (ensemble rule, threshold 0.5)
  stage 2  span located in the window the presence stage selected,
           token-F1 >= 0.5 against the gold clause
  stage 3  risk label correct           (fixed per-category lookup)

Stage 2 uses only inference-time information: the window is the argmax-scoring
window of the presence model, never the gold offset.
"""
import os, sys, re, json, string
from collections import Counter
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common
from common import NB

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RULE = sys.argv[1] if len(sys.argv) > 1 else "AND"      # AND | TRANS
FOCUS, QA_MAXLEN = 1200, 320

z = np.load(f"{HERE}/runs/base24k_seed42/probs.npz", allow_pickle=True)
pt, pf, Y = z["proba_trans"], z["proba_tfidf"], z["Y"]
cats, titles = list(z["cats"]), list(z["titles"])
D = common.load_all()
wins_by_title = {t: common.make_windows(D["ctx"][t]) for t in titles}

score = np.minimum(pf, pt) if RULE == "AND" else pt
detected = score >= 0.5

gold_idx = [(i, j) for i in range(len(titles)) for j in range(len(cats)) if Y[i, j] == 1]
det_gold = [(i, j) for (i, j) in gold_idx if detected[i, j]]
print(f"rule={RULE}  gold pairs={len(gold_idx)}  presence-detected={len(det_gold)}", flush=True)

# ---- stage 2: which window does the presence model pick for each detected pair?
tok = AutoTokenizer.from_pretrained(f"{NB}/outputs/presence_mil/final")
pres = AutoModelForSequenceClassification.from_pretrained(
    f"{NB}/outputs/presence_mil/final").to(DEVICE).eval()

pairs = []
for (i, j) in det_gold:
    for wi, w in enumerate(wins_by_title[titles[i]]):
        pairs.append((i, j, wi, D["cat_question"][cats[j]], w))
print(f"scoring {len(pairs)} windows to locate the selected window", flush=True)
probs = np.zeros(len(pairs)); B = 64
for k in range(0, len(pairs), B):
    ch = pairs[k:k + B]
    e = tok([c[3] for c in ch], [c[4] for c in ch], truncation=True,
            max_length=256, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        probs[k:k + B] = F.softmax(pres(**e).logits, -1)[:, 1].float().cpu().numpy()
    if (k // B) % 200 == 0: print(f"  {k}/{len(pairs)}", flush=True)

best = {}
for (i, j, wi, _, _), p in zip(pairs, probs):
    if (i, j) not in best or p > best[(i, j)][1]:
        best[(i, j)] = (wi, p)
del pres
if DEVICE == "mps": torch.mps.empty_cache()

# ---- run the span model inside the selected window (1200-char focus chunks)
qtok = AutoTokenizer.from_pretrained(f"{NB}/outputs/span/final")
qa = AutoModelForQuestionAnswering.from_pretrained(f"{NB}/outputs/span/final").to(DEVICE).eval()

norm = lambda s: " ".join("".join(c for c in re.sub(r"\b(a|an|the)\b", " ", s.lower())
                                  if c not in string.punctuation).split())
def token_f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g: return float(p == g)
    n = sum((Counter(p) & Counter(g)).values())
    if n == 0: return 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc)

def extract(question, window):
    chunks = [window[s:s + FOCUS] for s in range(0, max(1, len(window) - FOCUS + 1), 800)] or [window]
    enc = qtok([question] * len(chunks), chunks, truncation="only_second",
               max_length=QA_MAXLEN, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = qa(**enc)
    sl, el = out.start_logits.cpu().numpy(), out.end_logits.cpu().numpy()
    ids = enc["input_ids"].cpu().numpy()
    bestc, besttxt = -1e18, ""
    for k in range(len(chunks)):
        row = list(ids[k])
        L = int(enc["attention_mask"][k].sum())
        sep = row.index(qtok.sep_token_id)
        mask = np.full(len(row), -1e9); mask[sep + 1:L] = 0.0
        s = int(np.argmax(sl[k] + mask)); e = int(np.argmax(el[k] + mask))
        if e < s: e = s
        e = min(e, s + 60)
        conf = float(sl[k][s] + el[k][e])
        if conf > bestc:
            bestc, besttxt = conf, qtok.decode(row[s:e + 1], skip_special_tokens=True)
    return besttxt

rows = []
for n, ((i, j), (wi, p)) in enumerate(best.items()):
    win = wins_by_title[titles[i]][wi]
    pred = extract(D["cat_question"][cats[j]], win)
    golds = D["gold_spans"].get((titles[i], cats[j]), [])
    f1 = max((token_f1(pred, g) for g in golds), default=0.0)
    rows.append(dict(title=titles[i], category=cats[j], window=int(wi),
                     pres_prob=float(p), token_f1=f1, span_ok=int(f1 >= 0.5)))
    if n % 200 == 0: print(f"  span {n}/{len(best)}", flush=True)

df = pd.DataFrame(rows)
df.to_json(f"{HERE}/e2e_rows_{RULE}.json", orient="records")

n_gold = len(gold_idx); n_pres = len(det_gold); n_span = int(df.span_ok.sum())
# stage 3: the risk level is a fixed per-category lookup, so it is correct
# exactly when the category is correct -- no additional attrition.
res = dict(rule=RULE, n_gold=n_gold, n_presence=n_pres, n_span_ok=n_span,
           n_end_to_end=n_span,
           presence_recall=n_pres / n_gold,
           span_given_presence=n_span / n_pres,
           end_to_end=n_span / n_gold,
           mean_token_f1_given_presence=float(df.token_f1.mean()))
# cluster bootstrap over contracts for the end-to-end rate
rng = np.random.default_rng(42)
gold_by_title = {t: [] for t in titles}
ok = {(r["title"], r["category"]) for r in rows if r["span_ok"]}
for (i, j) in gold_idx:
    gold_by_title[titles[i]].append(int((titles[i], cats[j]) in ok))
vals = [np.array(gold_by_title[t]) for t in titles if gold_by_title[t]]
bs = []
for _ in range(10000):
    pick = rng.integers(0, len(vals), len(vals))
    cat_ = np.concatenate([vals[p] for p in pick])
    bs.append(cat_.mean())
res["end_to_end_ci"] = [float(x) for x in np.percentile(bs, [2.5, 97.5])]
json.dump(res, open(f"{HERE}/e2e_{RULE}.json", "w"), indent=2)
for k, v in res.items(): print(k, v, flush=True)
