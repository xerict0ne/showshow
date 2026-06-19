import torch
import torch.nn.functional as F
import pandas as pd
from common import RESULTS_DIR, benchmark, get_device

def naive_attention(q, k, v):
    # q, k, v: [batch, heads, seq, head_dim]
    scale = q.size(-1) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

def main():
    device = get_device()
    print(f"Running attention benchmark on: {device}")

    batch = 1
    heads = 8
    seq = 512 if device == "cuda" else 128
    head_dim = 64
    dtype = torch.float16 if device == "cuda" else torch.float32

    q = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, heads, seq, head_dim, device=device, dtype=dtype)

    naive_time = benchmark(lambda: naive_attention(q, k, v), warmup=3, iters=10, device=device)

    try:
        sdpa_time = benchmark(
            lambda: F.scaled_dot_product_attention(q, k, v),
            warmup=3,
            iters=10,
            device=device
        )
        sdpa_note = "PyTorch SDPA"
    except Exception as e:
        sdpa_time = naive_time
        sdpa_note = f"SDPA unavailable, fallback to naive: {e}"

    rows = [
        {"method": "Naive Attention", "time_sec": naive_time, "relative_time": 100.0, "note": "explicit QK^T + softmax + V"},
        {"method": "PyTorch SDPA", "time_sec": sdpa_time, "relative_time": sdpa_time / naive_time * 100, "note": sdpa_note}
    ]

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "attention_results.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)
    print(df)

if __name__ == "__main__":
    main()
