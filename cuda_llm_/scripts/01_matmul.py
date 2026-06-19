import torch
import pandas as pd
from common import RESULTS_DIR, benchmark, get_device, cuda_sync

def matmul_time(n: int, device: str, iters: int = 10):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    def fn():
        return torch.matmul(a, b)

    return benchmark(fn, warmup=5, iters=iters, device=device)


def main():
    sizes = [512, 1024, 2048]
    # 4096 can be added on RTX 3060/3070, but 2048 is safer for laptops.
    gpu_device = get_device()
    rows = []

    print(f"Using device for GPU-side test: {gpu_device}")

    for n in sizes:
        print(f"\nMatrix size: {n} x {n}")

        cpu_t = matmul_time(n, "cpu", iters=5)
        print(f"CPU avg time: {cpu_t:.6f} s")

        if gpu_device == "cuda":
            gpu_t = matmul_time(n, "cuda", iters=10)
            print(f"GPU avg time: {gpu_t:.6f} s")
        else:
            gpu_t = cpu_t
            print("CUDA unavailable. GPU time is set to CPU fallback time.")

        speedup = cpu_t / gpu_t if gpu_t > 0 else 0

        rows.append({
            "size": n,
            "cpu_time_sec": cpu_t,
            "gpu_time_sec": gpu_t,
            "speedup": speedup,
            "device": gpu_device
        })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "matmul_results.csv"
    df.to_csv(out, index=False)
    print("\nSaved:", out)
    print(df)

if __name__ == "__main__":
    main()
