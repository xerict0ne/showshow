import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from common import RESULTS_DIR, get_device, cuda_sync, reset_cuda_memory, get_peak_memory_mb

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=1024, hidden=4096, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_once(mode: str, device: str, steps: int = 100):
    model = SimpleMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = 256 if device == "cuda" else 64
    x = torch.randn(batch_size, 1024, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    reset_cuda_memory(device)
    cuda_sync(device)

    start = time.time()

    if mode == "FP32 Training" or device != "cuda":
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    elif mode == "AMP Training":
        # New PyTorch style AMP API.
        scaler = torch.amp.GradScaler("cuda")
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    cuda_sync(device)
    epoch_time = time.time() - start
    gpu_memory = get_peak_memory_mb(device)

    return epoch_time, gpu_memory


def main():
    device = get_device()
    print(f"Running AMP benchmark on: {device}")

    steps = 100 if device == "cuda" else 30

    fp32_time, fp32_mem = train_once("FP32 Training", device, steps)
    amp_time, amp_mem = train_once("AMP Training", device, steps)

    # On CPU fallback, AMP is not meaningful. Keep note clear.
    rows = [
        {
            "mode": "FP32 Training",
            "epoch_time_sec": fp32_time,
            "gpu_memory_mb": fp32_mem,
            "relative_epoch_time": 100.0,
            "relative_gpu_memory": 100.0 if fp32_mem > 0 else 0.0,
            "note": "baseline"
        },
        {
            "mode": "AMP Training",
            "epoch_time_sec": amp_time,
            "gpu_memory_mb": amp_mem,
            "relative_epoch_time": amp_time / fp32_time * 100,
            "relative_gpu_memory": amp_mem / fp32_mem * 100 if fp32_mem > 0 else 0.0,
            "note": "AMP is meaningful on CUDA; CPU fallback only validates code path"
        }
    ]

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "amp_results.csv"
    df.to_csv(out, index=False)
    print("Saved:", out)
    print(df)

if __name__ == "__main__":
    main()
