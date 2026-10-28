import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=15).decode("utf-8", errors="ignore").strip()
    except Exception as exc:
        return f"error: {exc}"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_b, used_b, free_b = shutil.disk_usage(str(root))
    hf_token_path = Path.home() / ".cache" / "huggingface" / "token"

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disk_root": {
            "total_gib": round(total_b / (1024 ** 3), 2),
            "used_gib": round(used_b / (1024 ** 3), 2),
            "free_gib": round(free_b / (1024 ** 3), 2),
        },
        "gpu_smi": run_cmd("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader"),
        "gpu_processes": run_cmd("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"),
        "memory_free_h": run_cmd("free -h"),
        "tools": {
            "python": shutil.which("python") or shutil.which("python3"),
            "ffmpeg": shutil.which("ffmpeg"),
            "tmux": shutil.which("tmux"),
            "nvcc": shutil.which("nvcc"),
        },
        "hf_token_file_present": hf_token_path.exists(),
    }

    try:
        import torch
        import torchaudio
        import k2
        import lhotse
        import sherpa_onnx

        report["python_packages"] = {
            "torch": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torchaudio": torchaudio.__version__,
            "k2": getattr(k2, "__dev_version__", getattr(k2, "__version__", "installed")),
            "lhotse": lhotse.__version__,
            "sherpa_onnx": getattr(sherpa_onnx, "__version__", "installed"),
        }
    except Exception as exc:
        report["python_packages_error"] = str(exc)

    out_path = reports_dir / "preflight_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote preflight report to {out_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
