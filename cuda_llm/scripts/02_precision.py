import torch
import pandas as pd
from common import RESULTS_DIR, benchmark, get_device

def run_matmul(dtype, device, n=2048, iters=10):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    def fn():
        return torch.matmul(a, b)

    return benchmark(fn, warmup=5, iters=iters, device=device)


def run_int8_sim(device, n=2048, iters=10):
    # INT8-sim is not TensorRT INT8 kernel. It simulates quantize/dequantize behavior.
    x = torch.randn(n, n, device=device)
    w = torch.randn(n, n, device=device)

    def fake_quant(t):
        scale = t.abs().max() / 127
        q = torch.clamp((t / scale).round(), -128, 127).to(torch.int8)
        return q.float() * scale

    def fn():
        xq = fake_quant(x)
        wq = fake_quant(w)
        return torch.matmul(xq, wq)

    return benchmark(fn, warmup=2, iters=max(3, iters // 2), device=device)


def main():
    device = get_device()
    print(f"Running precision benchmark on: {device}")

    rows = []

    fp32_time = run_matmul(torch.float32, device, n=1024 if device == "cpu" else 2048)
    rows.append({"precision": "FP32", "latency_sec": fp32_time, "relative_latency": 100.0, "note": "baseline"})

    if device == "cuda":
        fp16_time = run_matmul(torch.float16, device, n=2048)
        int8_time = run_int8_sim(device, n=1024)
    else:
        # CPU FP16 matmul can be slower, so keep a smaller safe run.
        fp16_time = run_matmul(torch.float32, device, n=1024)
        int8_time = run_int8_sim(device, n=512)

    rows.append({
        "precision": "FP16",
        "latency_sec": fp16_time,
        "relative_latency": fp16_time / fp32_time * 100,
        "note": "half precision on CUDA; CPU fallback if no CUDA"
    })
    rows.append({
        "precision": "INT8-sim",
        "latency_sec": int8_time,
        "relative_latency": int8_time / fp32_time * 100,
        "note": "simulation, not real TensorRT INT8 kernel"
    })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "precision_results.csv"
    df.to_csv(out, index=False)
    print("\nSaved:", out)
    print(df)

if __name__ == "__main__":
    main()
