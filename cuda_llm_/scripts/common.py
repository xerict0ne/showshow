from pathlib import Path
import time
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def cuda_sync(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_memory(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb(device: str) -> float:
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0


def benchmark(fn, warmup=5, iters=10, device="cpu"):
    for _ in range(warmup):
        fn()
    cuda_sync(device)

    start = time.time()
    for _ in range(iters):
        fn()
    cuda_sync(device)

    return (time.time() - start) / iters
