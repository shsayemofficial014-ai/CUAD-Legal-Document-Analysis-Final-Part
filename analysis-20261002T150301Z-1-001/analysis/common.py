"""Shared data prep for the supervisor-requested experiments.

Loads CUAD from the local cache (no network), rebuilds the exact 80/20
contract-level split (seed 42) and the exact windowing scheme used in the
thesis, so every experiment here is comparable with the reported numbers.
"""
import os, json, random, pickle
from collections import defaultdict
import numpy as np, pandas as pd

NB   = os.path.expanduser("~/Desktop/final project p3/notebooks")
CUAD = os.path.expanduser("~/p3_notebook/CUADv1.json")
SEED = 42
WIN, STRIDE = 2000, 1500


def load_all():
    js = json.load(open(CUAD))["data"]
    cat_from_qid = lambda q: q.rsplit("__", 1)[-1]
    titles = [c["title"] for c in js]

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(titles))
    test_set = {titles[i] for i in perm[:int(0.20 * len(titles))]}
    train_titles = [t for t in titles if t not in test_set]
    test_titles  = [t for t in titles if t in test_set]

    ctx, presence = {}, []
    for c in js:
        t = c["title"]; para = c["paragraphs"][0]; ctx[t] = para["context"]
        for qa in para["qas"]:
            presence.append({"title": t, "category": cat_from_qid(qa["id"]),
                             "question": qa["question"],
                             "label": 1 if qa["answers"] else 0})
    pres_df = pd.DataFrame(presence)
    cats = sorted(pres_df["category"].unique())
    cat_question = pres_df.groupby("category")["question"].first().to_dict()

    gold_spans = defaultdict(list)
    for c in js:
        for qa in c["paragraphs"][0]["qas"]:
            for a in qa["answers"]:
                if a["text"].strip():
                    gold_spans[(c["title"], cat_from_qid(qa["id"]))].append(a["text"])

    return dict(js=js, titles=titles, train_titles=train_titles,
                test_titles=test_titles, ctx=ctx, cats=cats,
                cat_question=cat_question, gold_spans=gold_spans,
                pres_df=pres_df)


def make_windows(text):
    wins, pos = [], 0
    while pos < len(text):
        wins.append(text[pos:pos + WIN])
        if pos + WIN >= len(text):
            break
        pos += STRIDE
    return wins or [text]


def build_window_examples(D, seed=SEED, neg_per_bag=2):
    """Exactly the Cell-17 construction: positives = every window containing a
    gold span; negatives = `neg_per_bag` sampled windows per bag."""
    _random = random.Random(seed)
    tr = set(D["train_titles"])
    has = lambda spans, w: any(s[:80] in w for s in spans)
    rows = []
    for c in D["js"]:
        t = c["title"]
        if t not in tr:
            continue
        wins = make_windows(D["ctx"][t])
        for qa in c["paragraphs"][0]["qas"]:
            cat = qa["id"].rsplit("__", 1)[-1]
            spans = D["gold_spans"].get((t, cat), [])
            if spans:
                pos_idx = [i for i, w in enumerate(wins) if has(spans, w)]
                neg_idx = [i for i in range(len(wins)) if i not in pos_idx]
                for i in pos_idx:
                    rows.append({"question": qa["question"], "window": wins[i], "label": 1})
                for i in _random.sample(neg_idx, min(neg_per_bag, len(neg_idx))):
                    rows.append({"question": qa["question"], "window": wins[i], "label": 0})
            else:
                for i in _random.sample(range(len(wins)), min(neg_per_bag, len(wins))):
                    rows.append({"question": qa["question"], "window": wins[i], "label": 0})
    return pd.DataFrame(rows)


def tfidf_probs(D, te_titles):
    bl = pickle.load(open(f"{NB}/artifacts/baseline.pkl", "rb"))
    vec, clf, cats = bl["vec"], bl["classifiers"], list(bl["cats"])
    X = vec.transform([D["ctx"][t] for t in te_titles])
    P = np.zeros((len(te_titles), len(cats)))
    for j, cat in enumerate(cats):
        kind, obj = clf[cat]
        P[:, j] = float(obj) if kind == "const" else obj.predict_proba(X)[:, 1]
    return P, cats


def labels(D, te_titles, cats):
    return np.array([[int(bool(D["gold_spans"].get((t, c), []))) for c in cats]
                     for t in te_titles])


def micro_f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
