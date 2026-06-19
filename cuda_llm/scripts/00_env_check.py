import torch
import platform
from common import RESULTS_DIR

def main():
    lines = []
    lines.append("CUDA LLM Heterogeneous Computing Lab - Environment Check")
    lines.append("=" * 70)
    lines.append(f"Python platform: {platform.platform()}")
    lines.append(f"PyTorch version: {torch.__version__}")
    lines.append(f"CUDA available: {torch.cuda.is_available()}")
    lines.append(f"CUDA version in PyTorch: {torch.version.cuda}")

    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        lines.append(f"GPU name: {torch.cuda.get_device_name(idx)}")
        lines.append(f"GPU capability: {prop.major}.{prop.minor}")
        lines.append(f"GPU total memory: {prop.total_memory / 1024**3:.2f} GB")
    else:
        lines.append("GPU name: N/A")
        lines.append("GPU capability: N/A")
        lines.append("GPU total memory: N/A")

    text = "\n".join(lines)
    print(text)

    out = RESULTS_DIR / "env_info.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
