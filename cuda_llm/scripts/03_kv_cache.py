import pandas as pd
from common import RESULTS_DIR

def estimate_kv_cache_gb(seq_len, num_layers=32, num_heads=32, head_dim=128, batch_size=1, bytes_per_value=2):
    # KV cache stores both K and V:
    # batch * seq_len * layers * heads * head_dim * 2(K,V) * bytes
    total_bytes = batch_size * seq_len * num_layers * num_heads * head_dim * 2 * bytes_per_value
    return total_bytes / 1024**3

def main():
    seq_lens = [512, 1024, 2048, 4096, 8192, 16384]

    rows = []
    for s in seq_lens:
        mem_gb = estimate_kv_cache_gb(s)
        rows.append({
            "seq_len": s,
            "kv_cache_gb": mem_gb,
            "num_layers": 32,
            "num_heads": 32,
            "head_dim": 128,
            "batch_size": 1,
            "dtype": "FP16/BF16"
        })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "kv_cache_results.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)
    print(df)

if __name__ == "__main__":
    main()
