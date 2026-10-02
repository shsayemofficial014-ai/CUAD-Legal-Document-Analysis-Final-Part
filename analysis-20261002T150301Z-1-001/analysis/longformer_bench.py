"""Back the Section 2.3 sparse-attention argument with measurements instead of
assertions.

Longformer-base is instantiated from its published configuration (12 layers,
768 hidden, 4,096-token window, attention window 512) with randomly initialised
weights: forward/backward cost and memory depend on the shapes, not the values,
so no pretrained download is needed.

Reports, on this laptop:
  * parameter count vs DistilBERT
  * forward latency at 4,096 tokens vs DistilBERT at 256
  * peak memory for one training step (forward + backward + AdamW) at batch 1
  * projected cost of scanning one median contract for all 41 categories
"""
import json, time, gc, os, sys
import torch
from transformers import (LongformerConfig, LongformerForSequenceClassification,
                          DistilBertConfig, DistilBertForSequenceClassification)

HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", dev, flush=True)

def mps_peak_reset():
    if dev == "mps":
        torch.mps.empty_cache()
        try: torch.mps.reset_peak_memory_stats()
        except Exception: pass

def mps_peak():
    if dev != "mps": return None
    try: return torch.mps.driver_allocated_memory() / 1e9
    except Exception: return None

def bench(model, seq, batch, label, train_step=False, reps=3):
    gc.collect(); mps_peak_reset()
    model = model.to(dev)
    ids = torch.randint(0, 1000, (batch, seq), device=dev)
    am = torch.ones_like(ids)
    out = {"seq": seq, "batch": batch}
    try:
        if train_step:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            labels = torch.zeros(batch, dtype=torch.long, device=dev)
            t0 = time.time()
            for _ in range(reps):
                opt.zero_grad()
                loss = model(input_ids=ids, attention_mask=am, labels=labels).loss
                loss.backward(); opt.step()
            if dev == "mps": torch.mps.synchronize()
            out["sec_per_step"] = (time.time() - t0) / reps
        else:
            model.eval()
            with torch.no_grad():
                model(input_ids=ids, attention_mask=am)
                if dev == "mps": torch.mps.synchronize()
                t0 = time.time()
                for _ in range(reps):
                    model(input_ids=ids, attention_mask=am)
                if dev == "mps": torch.mps.synchronize()
            out["sec_per_forward"] = (time.time() - t0) / reps
        out["peak_gb"] = mps_peak()
        out["ok"] = True
    except Exception as e:
        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    del model, ids, am
    gc.collect(); mps_peak_reset()
    return out

lf_cfg = LongformerConfig(num_labels=2)
db_cfg = DistilBertConfig(num_labels=2)
lf = LongformerForSequenceClassification(lf_cfg)
db = DistilBertForSequenceClassification(db_cfg)
res["params"] = {"longformer_base": sum(p.numel() for p in lf.parameters()),
                 "distilbert_base": sum(p.numel() for p in db.parameters())}
res["config"] = {"longformer": {"layers": lf_cfg.num_hidden_layers,
                                "hidden": lf_cfg.hidden_size,
                                "max_pos": lf_cfg.max_position_embeddings,
                                "attention_window": lf_cfg.attention_window[0]},
                 "distilbert": {"layers": db_cfg.n_layers, "hidden": db_cfg.dim,
                                "max_pos": db_cfg.max_position_embeddings}}
print(json.dumps(res, indent=1), flush=True)

print("\n-- inference --", flush=True)
res["lf_forward_4096_b1"] = bench(lf, 4096, 1, "lf-fwd")
print("longformer fwd 4096:", res["lf_forward_4096_b1"], flush=True)
res["db_forward_256_b1"]  = bench(db, 256, 1, "db-fwd")
print("distilbert fwd 256 :", res["db_forward_256_b1"], flush=True)
res["db_forward_256_b16"] = bench(db, 256, 16, "db-fwd16")
print("distilbert fwd 256 b16:", res["db_forward_256_b16"], flush=True)

print("\n-- training step (fwd+bwd+AdamW) --", flush=True)
lf = LongformerForSequenceClassification(lf_cfg)
res["lf_train_4096_b1"] = bench(lf, 4096, 1, "lf-train", train_step=True, reps=2)
print("longformer train b1:", res["lf_train_4096_b1"], flush=True)
lf = LongformerForSequenceClassification(lf_cfg)
res["lf_train_4096_b2"] = bench(lf, 4096, 2, "lf-train2", train_step=True, reps=2)
print("longformer train b2:", res["lf_train_4096_b2"], flush=True)
db = DistilBertForSequenceClassification(db_cfg)
res["db_train_256_b16"] = bench(db, 256, 16, "db-train", train_step=True, reps=3)
print("distilbert train b16:", res["db_train_256_b16"], flush=True)

# projected cost of one median contract (33,143 chars) across 41 categories
MEDIAN_CHARS = 33143
lf_wins = -(-int(MEDIAN_CHARS / 4) // 4096)          # ~4 chars/token, 4096-token windows
db_wins = 22                                          # 2,000-char windows, 1,500 stride
if res["lf_forward_4096_b1"].get("ok"):
    res["contract_scan"] = {
        "longformer_windows": lf_wins,
        "distilbert_windows": db_wins,
        "longformer_sec_41_cats": lf_wins * 41 * res["lf_forward_4096_b1"]["sec_per_forward"],
        "distilbert_sec_41_cats": db_wins * 41 * res["db_forward_256_b16"]["sec_per_forward"] / 16,
    }
json.dump(res, open(f"{HERE}/longformer_bench.json", "w"), indent=2)
print("\nwrote longformer_bench.json", flush=True)
