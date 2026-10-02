"""Latency benchmark of the deployed Clausify pipeline, using the application's
own inference code (model_utils.analyze) rather than a re-implementation."""
import os, sys, time, json, resource
import numpy as np
APP = os.path.expanduser("~/Desktop/final project app p3")
sys.path.insert(0, APP); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CLAUSIFY_MODELS", os.path.expanduser("~/Desktop/final project p3/notebooks/outputs"))
import common
import model_utils as mu

D = common.load_all()
te = D["test_titles"]
lens = sorted((len(D["ctx"][t]), t) for t in te)
picks = {"shortest": lens[0], "25th pct": lens[len(lens)//4], "median": lens[len(lens)//2],
         "75th pct": lens[3*len(lens)//4], "longest": lens[-1]}
print("models available:", mu.models_available(), flush=True)
t0 = time.time(); mu.warm_up(); warm = time.time()-t0
print(f"model load / warm-up: {warm:.1f} s", flush=True)

rows = []
for name, (n, t) in picks.items():
    text = D["ctx"][t]
    nwin = len(mu.make_windows(text))
    t0 = time.time(); res = mu.analyze(text, threshold=0.5); dt = time.time()-t0
    detected = len(res["detected"]) if isinstance(res, dict) and "detected" in res else None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9
    rows.append(dict(bucket=name, chars=n, windows_after_cap=min(nwin, mu.MAX_WINDOWS),
                     windows_uncapped=nwin, seconds=dt, detected=detected, peak_rss_gb=rss))
    print(f"  {name:9s} {n:7,d} chars  {nwin:3d} windows (capped to {min(nwin,mu.MAX_WINDOWS)})  "
          f"{dt:6.1f} s  detected={detected}  peakRSS={rss:.2f} GB", flush=True)

secs = [r["seconds"] for r in rows]
out = dict(warm_up_seconds=warm, rows=rows, mean=float(np.mean(secs)), median=float(np.median(secs)),
           min=float(min(secs)), max=float(max(secs)),
           peak_rss_gb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9),
           window_cap=mu.MAX_WINDOWS, device="cpu")
json.dump(out, open(f"{os.path.dirname(os.path.abspath(__file__))}/app_bench.json","w"), indent=2)
print(f"\nmean {np.mean(secs):.1f} s | median {np.median(secs):.1f} s | max {max(secs):.1f} s")
