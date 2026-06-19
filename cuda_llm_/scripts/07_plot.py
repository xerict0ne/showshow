from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from common import RESULTS_DIR, FIGURES_DIR

def savefig(name):
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    print("Saved:", path)

def plot_matmul():
    df = pd.read_csv(RESULTS_DIR / "matmul_results.csv")
    plt.figure(figsize=(7, 4.5))
    plt.bar(df["size"].astype(str), df["speedup"])
    plt.xlabel("Matrix size N × N")
    plt.ylabel("Speedup (CPU time / GPU time)")
    plt.title("CPU vs GPU GEMM Speedup")
    savefig("01_cpu_gpu_gemm_speedup.png")

def plot_precision():
    df = pd.read_csv(RESULTS_DIR / "precision_results.csv")
    plt.figure(figsize=(7, 4.5))
    plt.bar(df["precision"], df["relative_latency"])
    plt.xlabel("Precision mode")
    plt.ylabel("Relative latency (%)")
    plt.title("Precision and Quantization Latency Comparison")
    savefig("02_precision_latency_comparison.png")

def plot_kv_cache():
    df = pd.read_csv(RESULTS_DIR / "kv_cache_results.csv")
    plt.figure(figsize=(7, 4.5))
    plt.plot(df["seq_len"], df["kv_cache_gb"], marker="o")
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("Estimated KV Cache memory (GB)")
    plt.title("KV Cache Memory Growth with Context Length")
    savefig("03_kv_cache_growth.png")

def plot_attention():
    df = pd.read_csv(RESULTS_DIR / "attention_results.csv")
    plt.figure(figsize=(7, 4.5))
    plt.bar(df["method"], df["relative_time"])
    plt.xlabel("Attention implementation")
    plt.ylabel("Relative execution time (%)")
    plt.title("Attention Optimization: Naive vs SDPA")
    savefig("04_attention_naive_vs_sdpa.png")

def plot_amp():
    df = pd.read_csv(RESULTS_DIR / "amp_results.csv")
    x = list(range(len(df)))
    width = 0.35

    plt.figure(figsize=(7, 4.5))
    plt.bar([i - width / 2 for i in x], df["relative_epoch_time"], width, label="Epoch time")
    plt.bar([i + width / 2 for i in x], df["relative_gpu_memory"], width, label="GPU memory")
    plt.xticks(x, df["mode"])
    plt.ylabel("Relative value (%)")
    plt.title("AMP Mixed Precision Training Benefit")
    plt.legend()
    savefig("05_amp_training_speed_memory.png")

def plot_profile():
    df = pd.read_csv(RESULTS_DIR / "profile_results.csv")
    df = df.sort_values("cuda_time_share", ascending=True)

    plt.figure(figsize=(7, 4.5))
    plt.barh(df["operation"], df["cuda_time_share"])
    plt.xlabel("CUDA time share (%)")
    plt.title("Profiling Breakdown of LLM-style Workload")
    savefig("06_profiling_breakdown.png")

def main():
    plot_matmul()
    plot_precision()
    plot_kv_cache()
    plot_attention()
    plot_amp()
    plot_profile()

if __name__ == "__main__":
    main()
