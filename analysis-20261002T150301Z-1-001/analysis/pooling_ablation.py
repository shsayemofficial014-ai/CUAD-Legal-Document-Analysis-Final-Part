"""Ablation over the aggregation rule.

Section 4.2 chose max-pooling because it implements the MIL decision rule, and
Section 6.1 blamed max-pooling for the transformer's inflated false positives.
Neither claim was tested against alternatives. This scores every window once and
then compares aggregation functions on identical window scores, so the only
thing that varies is the rule.
"""
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import common; from common import NB
DEV="mps" if torch.backends.mps.is_available() else "cpu"

D=common.load_all(); te=D["test_titles"]
pf,cats=common.tfidf_probs(D,te); Y=common.labels(D,te,cats)
wins={t:common.make_windows(D["ctx"][t]) for t in te}
CACHE=f"{HERE}/window_scores.npz"

if os.path.exists(CACHE):
    d=np.load(CACHE,allow_pickle=True); W=d["W"].item()
    print("loaded cached window scores")
else:
    tok=AutoTokenizer.from_pretrained(f"{NB}/outputs/presence_mil/final")
    m=AutoModelForSequenceClassification.from_pretrained(f"{NB}/outputs/presence_mil/final").to(DEV).eval()
    rows=[(i,j,D["cat_question"][c],w) for i,t in enumerate(te) for j,c in enumerate(cats) for w in wins[t]]
    print("scoring",len(rows),"pairs",flush=True)
    probs=np.zeros(len(rows)); B=64; t0=time.time()
    for k in range(0,len(rows),B):
        ch=rows[k:k+B]
        e=tok([r[2] for r in ch],[r[3] for r in ch],truncation=True,max_length=256,padding=True,return_tensors="pt").to(DEV)
        with torch.no_grad(): probs[k:k+B]=F.softmax(m(**e).logits,-1)[:,1].float().cpu().numpy()
        if (k//B)%400==0:
            el=(time.time()-t0)/60; print(f"  {k}/{len(rows)} {el:.1f}m eta {el/max(k,1)*(len(rows)-k):.1f}m",flush=True)
    W={}
    for (i,j,_,_),p in zip(rows,probs): W.setdefault((i,j),[]).append(float(p))
    np.savez_compressed(CACHE,W=np.array(W,dtype=object))
    print("scored in",(time.time()-t0)/60,"min",flush=True)

n_t,n_c=len(te),len(cats)
def agg(fn):
    M=np.zeros((n_t,n_c))
    for (i,j),v in W.items(): M[i,j]=fn(np.array(v))
    return M
def topk_mean(k):
    return lambda v: float(np.sort(v)[::-1][:k].mean())
def noisy_or(v): return float(1-np.prod(1-np.clip(v,0,0.999999)))

POOLS={"max (thesis)":lambda v:float(v.max()),
       "mean":lambda v:float(v.mean()),
       "top-3 mean":topk_mean(3),
       "top-5 mean":topk_mean(5),
       "noisy-OR":noisy_or,
       "2nd highest":lambda v:float(np.sort(v)[::-1][min(1,len(v)-1)])}

def micro(y,p):
    tp=int(((y==1)&(p==1)).sum());fp_=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
    if tp==0: return 0.0,0.0,0.0
    pr,rc=tp/(tp+fp_),tp/(tp+fn); return 2*pr*rc/(pr+rc),pr,rc

ths=np.round(np.arange(0.05,0.96,0.05),2)
out={}
print("\n=== aggregation ablation (transformer alone, and AND with TF-IDF) ===")
print(f"{'pooling':14s} {'F1@0.5':>7s} {'P@0.5':>6s} {'R@0.5':>6s} {'bestF1':>7s} {'@th':>5s} {'AND F1':>7s}")
for name,fn in POOLS.items():
    M=agg(fn)
    f1,pr,rc=micro(Y,(M>=.5).astype(int))
    best=max(((micro(Y,(M>=t).astype(int))[0],t) for t in ths))
    andf1=micro(Y,(np.minimum(pf,M)>=.5).astype(int))[0]
    out[name]=dict(f1_at_05=f1,precision_at_05=pr,recall_at_05=rc,
                   best_f1=best[0],best_threshold=float(best[1]),and_f1_at_05=andf1)
    print(f"{name:14s} {f1:7.4f} {pr:6.3f} {rc:6.3f} {best[0]:7.4f} {best[1]:5.2f} {andf1:7.4f}")
out["tfidf_baseline_f1"]=micro(Y,(pf>=.5).astype(int))[0]
print(f"{'TF-IDF':14s} {out['tfidf_baseline_f1']:7.4f}")
json.dump(out,open(f"{HERE}/pooling_ablation.json","w"),indent=2)
print("\nwrote pooling_ablation.json")
