import json
from pathlib import Path
import subprocess


def run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=10).decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        return f"error: {exc}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    status_file = root / "reports" / "training_live_status.json"
    print("=== TMUX SESSIONS ===")
    print(run("tmux ls"))
    print("\n=== GPU STATUS ===")
    print(run("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"))
    print(run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"))
    print("\n=== DISK FREE ===")
    print(run("df -h /"))
    if status_file.exists():
        print("\n=== LIVE TRAINING STATUS JSON ===")
        print(status_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
