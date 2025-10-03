from __future__ import annotations
from pathlib import Path
import random, json, numpy as np
from datetime import datetime

def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    return {
        "root": root,
        "data": root / "data",
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "outputs": root / "outputs",
        "models": root / "models",
        "notebooks": root / "notebooks",
        "src": root / "src",
    }

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def load_json(path: Path):
    return json.loads(Path(path).read_text())

def timestamp(prefix: str = "exp") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
