"""Pull concrete false-positive, false-negative and span-error examples with
their actual text, so Chapter 6 can show what the numbers mean."""
import os, sys, json, re, string
from collections import Counter
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import common; from common import NB
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

z=np.load(f"{HERE}/runs/base24k_seed42/probs.npz",allow_pickle=True)
pt,pf,Y=z["proba_trans"],z["proba_tfidf"],z["Y"]; cats=list(z["cats"]); titles=list(z["titles"])
D=common.load_all(); wins={t:common.make_windows(D["ctx"][t]) for t in titles}
and_p=(np.minimum(pf,pt)>=.5).astype(int)

fp=[(i,j) for i in range(len(titles)) for j in range(len(cats)) if Y[i,j]==0 and and_p[i,j]==1]
fn=[(i,j) for i in range(len(titles)) for j in range(len(cats)) if Y[i,j]==1 and and_p[i,j]==0]
# rank FPs by confidence (most confident mistakes are the interesting ones)
fp.sort(key=lambda k: -min(pf[k],pt[k])); fn.sort(key=lambda k: min(pf[k],pt[k]))
print(f"false positives: {len(fp)}   false negatives: {len(fn)}")

tok=AutoTokenizer.from_pretrained(f"{NB}/outputs/presence_mil/final")
pres=AutoModelForSequenceClassification.from_pretrained(f"{NB}/outputs/presence_mil/final").to(DEV).eval()

def firing_window(i,j):
    q=D["cat_question"][cats[j]]; ws=wins[titles[i]]
    best=(-1,0)
    for k in range(0,len(ws),32):
        ch=ws[k:k+32]
        e=tok([q]*len(ch),ch,truncation=True,max_length=256,padding=True,return_tensors="pt").to(DEV)
        with torch.no_grad(): p=F.softmax(pres(**e).logits,-1)[:,1].float().cpu().numpy()
        m=int(p.argmax())
        if p[m]>best[0]: best=(float(p[m]),k+m)
    return best

out={"false_positives":[],"false_negatives":[]}
for (i,j) in fp[:4]:
    score,widx=firing_window(i,j); w=" ".join(wins[titles[i]][widx].split())
    out["false_positives"].append(dict(contract=titles[i],category=cats[j],
        tfidf=float(pf[i,j]),trans=float(pt[i,j]),window_index=widx,window_score=score,
        window_text=w[:600]))
    print(f"\n[FP] {cats[j]} | tfidf={pf[i,j]:.2f} trans={pt[i,j]:.2f} | window {widx} score {score:.2f}")
    print("    ", w[:330])
for (i,j) in fn[:4]:
    g=D["gold_spans"].get((titles[i],cats[j]),[])
    out["false_negatives"].append(dict(contract=titles[i],category=cats[j],
        tfidf=float(pf[i,j]),trans=float(pt[i,j]),
        gold_clause=" ".join(g[0].split())[:600] if g else ""))
    print(f"\n[FN] {cats[j]} | tfidf={pf[i,j]:.2f} trans={pt[i,j]:.2f}")
    print("     gold:", " ".join(g[0].split())[:300] if g else "")
del pres
if DEV=="mps": torch.mps.empty_cache()

# ---- span errors: run the QA model the way the pipeline does
qtok=AutoTokenizer.from_pretrained(f"{NB}/outputs/span/final")
qa=AutoModelForQuestionAnswering.from_pretrained(f"{NB}/outputs/span/final").to(DEV).eval()
norm=lambda s:" ".join("".join(c for c in re.sub(r"\b(a|an|the)\b"," ",s.lower()) if c not in string.punctuation).split())
def tf1(p,g):
    a,b=norm(p).split(),norm(g).split()
    if not a or not b: return float(a==b)
    n=sum((Counter(a)&Counter(b)).values())
    if n==0: return 0.0
    pr,rc=n/len(a),n/len(b); return 2*pr*rc/(pr+rc)
rows=json.load(open(f"{HERE}/e2e_rows_AND.json"))
bad=[r for r in rows if r["span_ok"]==0 and r["token_f1"]<0.3][:200]
picked=[]
for r in bad:
    t,c,widx=r["title"],r["category"],r["window"]
    w=wins[t][widx]; g=D["gold_spans"].get((t,c),[])
    if not g: continue
    chunks=[w[s:s+1200] for s in range(0,max(1,len(w)-1199),800)] or [w]
    e=qtok([D["cat_question"][c]]*len(chunks),chunks,truncation="only_second",max_length=320,padding=True,return_tensors="pt").to(DEV)
    with torch.no_grad(): o=qa(**e)
    sl,el=o.start_logits.cpu().numpy(),o.end_logits.cpu().numpy(); ids=e["input_ids"].cpu().numpy()
    bc,bt=-1e18,""
    for k in range(len(chunks)):
        row=list(ids[k]); L=int(e["attention_mask"][k].sum()); sep=row.index(qtok.sep_token_id)
        m=np.full(len(row),-1e9); m[sep+1:L]=0
        s_=int(np.argmax(sl[k]+m)); e_=int(np.argmax(el[k]+m))
        if e_<s_: e_=s_
        e_=min(e_,s_+60); conf=float(sl[k][s_]+el[k][e_])
        if conf>bc: bc,bt=conf,qtok.decode(row[s_:e_+1],skip_special_tokens=True)
    gold=max(g,key=len)
    win_has=any(x[:80] in w for x in g)
    picked.append(dict(contract=t,category=c,window_contains_gold=bool(win_has),
                       predicted=" ".join(bt.split())[:400],gold=" ".join(gold.split())[:400],
                       token_f1=tf1(bt,gold)))
    if len(picked)>=4: break
out["span_errors"]=picked
for p in picked:
    print(f"\n[SPAN] {p['category']} | window contains gold: {p['window_contains_gold']} | token-F1 {p['token_f1']:.2f}")
    print("   gold :", p["gold"][:230]); print("   pred :", p["predicted"][:230])
json.dump(out,open(f"{HERE}/error_examples.json","w"),indent=2)
print("\nwrote error_examples.json")
