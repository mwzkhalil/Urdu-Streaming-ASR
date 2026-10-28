import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party/icefall_recipe"))

runpy.run_path(
    str(ROOT / "scripts/train_transducer.py"),
    run_name="__main__",
)
