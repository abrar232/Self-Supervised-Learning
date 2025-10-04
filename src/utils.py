"""
Project utilities for the TF self-supervised repo: paths, seeding, tiny JSON I/O, timestamps.

What this provides
------------------
- `project_paths()` : common directories relative to the repo root
- `set_seed()`      : reproducible runs (Python, NumPy, TensorFlow)
- `ensure_dirs()`   : safe directory creation
- `save_json()` / `load_json()` : tiny JSON helpers
- `timestamp()`     : filesystem-friendly run/experiment names

Assumptions
-----------
This file lives at: <repo>/src/utils.py
So the repo root is computed as `Path(__file__).resolve().parents[1]`.

Quick usage
-----------
>>> from src.utils import project_paths, set_seed, timestamp
>>> P = project_paths(); set_seed(42); print(timestamp("ssl"))
"""
from __future__ import annotations
from pathlib import Path
import random, json, numpy as np
from datetime import datetime


def project_paths() -> dict[str, Path]:
    """
    Build a dictionary of common project paths relative to the **repo root**.

    Returns
    -------
    dict[str, Path]
        Keys: "root", "data", "raw", "processed", "outputs",
              "models", "notebooks", "src".
    """
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
    """
    Seed Python/NumPy and TensorFlow RNGs for better reproducibility.

    Parameters
    ----------
    seed : int, default=42
        Seed value used across RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow not installed or unavailable; ignore.
        pass


def ensure_dirs(*paths: Path) -> None:
    """
    Create directories if they don't exist (parents included).

    Example
    -------
    >>> from pathlib import Path
    >>> ensure_dirs(Path("outputs/figures"), Path("outputs/metrics"))
    """
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    """
    Save a Python object as pretty-printed JSON.

    Parameters
    ----------
    obj : Any
        JSON-serializable object (dict/list/str/num/etc.).
    path : Path
        Destination path; parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def load_json(path: Path):
    """
    Load a JSON file and return the parsed Python object.

    Parameters
    ----------
    path : Path
        Path to a JSON file.

    Returns
    -------
    Any
        Parsed JSON content (dict/list/...).
    """
    return json.loads(Path(path).read_text())


def timestamp(prefix: str = "exp") -> str:
    """
    Generate a filesystem-friendly timestamp string.

    Parameters
    ----------
    prefix : str, default="exp"
        Short prefix to prepend to the timestamp.

    Returns
    -------
    str
        e.g., "exp_2025-10-04_12-34-56"
    """
    return f"{prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
