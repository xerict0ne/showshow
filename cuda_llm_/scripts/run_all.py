import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

scripts = [
    "00_env_check.py",
    "01_matmul.py",
    "02_precision.py",
    "03_kv_cache.py",
    "04_attention.py",
    "05_amp.py",
    "06_profile.py",
    "07_plot.py",
]

def main():
    for name in scripts:
        print("\n" + "=" * 80)
        print(f"Running {name}")
        print("=" * 80)
        subprocess.run([sys.executable, str(SCRIPTS / name)], check=True)

if __name__ == "__main__":
    main()
