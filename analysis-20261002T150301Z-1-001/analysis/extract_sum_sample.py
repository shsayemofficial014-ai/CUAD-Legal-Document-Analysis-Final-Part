import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
D = common.load_all()
_test = set(D["test_titles"])
GOLD = json.load(open("gold_templates.json")) if os.path.exists("gold_templates.json") else None
rows = []
for c in D["js"]:
    if c["title"] not in _test: continue
    for qa in c["paragraphs"][0]["qas"]:
        cat = qa["id"].rsplit("__", 1)[-1]
        for a in qa["answers"]:
            at = a["text"].strip()
            if at:
                rows.append({"title": c["title"], "category": cat,
                             "input_text": f"summarize in plain English: {at[:1200]}",
                             "clause": at[:1200]})
df = pd.DataFrame(rows)
print("total test clauses:", len(df))
samp = df.sample(150, random_state=42).reset_index(drop=True)
samp.to_json("sum_sample150.json", orient="records", indent=1)
print(samp.category.value_counts().to_string())
