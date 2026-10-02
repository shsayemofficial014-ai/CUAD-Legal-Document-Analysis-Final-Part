"""Benchmark ONE configuration in its own process, so the peak MPS allocation
reported belongs to that configuration alone."""
import sys, json, time, torch
from transformers import (LongformerConfig, LongformerForSequenceClassification,
                          DistilBertConfig, DistilBertForSequenceClassification)

which, seq, batch, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
dev = "mps" if torch.backends.mps.is_available() else "cpu"

if which == "longformer":
    # published longformer-base-4096 geometry
    cfg = LongformerConfig(vocab_size=50265, max_position_embeddings=4098,
                           attention_window=[512] * 12, num_labels=2)
    model = LongformerForSequenceClassification(cfg)
else:
    cfg = DistilBertConfig(num_labels=2)
    model = DistilBertForSequenceClassification(cfg)

n_params = sum(p.numel() for p in model.parameters())
out = {"model": which, "seq": seq, "batch": batch, "mode": mode, "params": n_params,
       "device": dev}
try:
    model = model.to(dev)
    ids = torch.randint(0, 1000, (batch, seq), device=dev)
    am = torch.ones_like(ids)
    if mode == "train":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
        labels = torch.zeros(batch, dtype=torch.long, device=dev)
        for i in range(3):
            if i == 1:
                torch.mps.synchronize() if dev == "mps" else None
                t0 = time.time()
            opt.zero_grad()
            model(input_ids=ids, attention_mask=am, labels=labels).loss.backward()
            opt.step()
        if dev == "mps": torch.mps.synchronize()
        out["sec_per_step"] = (time.time() - t0) / 2
    else:
        model.eval()
        with torch.no_grad():
            model(input_ids=ids, attention_mask=am)
            if dev == "mps": torch.mps.synchronize()
            t0 = time.time()
            for _ in range(5):
                model(input_ids=ids, attention_mask=am)
            if dev == "mps": torch.mps.synchronize()
        out["sec_per_forward"] = (time.time() - t0) / 5
    out["ok"] = True
except Exception as e:
    out["ok"] = False
    out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
if dev == "mps":
    out["peak_gb"] = torch.mps.driver_allocated_memory() / 1e9
print(json.dumps(out))
