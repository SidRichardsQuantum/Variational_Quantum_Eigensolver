"""
vqe.io_utils
------------
Utility functions for reproducible VQE runs:
- Run configuration hashing
- JSON-safe serialization
- File/directory management
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict


# ================================================================
# HELPER FUNCTIONS
# ================================================================
def _round_floats(x: Any, ndigits: int = 8):
    """Recursively round floats, numpy arrays, or lists for stable hashing."""
    if isinstance(x, float):
        return round(x, ndigits)

    # Handle scalar numpy-like values
    try:
        if hasattr(x, "item") and isinstance(x.item(), float):
            return round(float(x), ndigits)
    except Exception:
        pass

    # Convert array-like to list and recurse
    if hasattr(x, "tolist"):
        return _round_floats(x.tolist(), ndigits)

    # Handle containers
    if isinstance(x, (list, tuple)):
        return type(x)(_round_floats(v, ndigits) for v in x)

    return x


def _to_serializable(obj: Any):
    """Recursively convert tensors, numpy arrays, or complex objects to JSON-serializable Python types."""
    if hasattr(obj, "item"):
        try:
            return float(obj.item())
        except Exception:
            pass

    if hasattr(obj, "tolist"):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    return obj


# ================================================================
# RUN CONFIGURATION & HASHING
# ================================================================
def make_run_config_dict(
    symbols,
    coordinates,
    basis: str,
    ansatz_desc: str,
    optimizer_name: str,
    stepsize: float,
    max_iterations: int,
    seed: int,
    mapping: str,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
) -> Dict[str, Any]:
    """
    Construct a canonical dictionary describing a VQE run configuration.

    This structure is used to create stable, hashable signatures for caching.

    Args:
        symbols: List of atomic symbols (e.g., ["H", "H"])
        coordinates: Molecular coordinates array
        basis: Basis set name (e.g., "sto-3g")
        ansatz_desc: Ansatz description or name
        optimizer_name: Name of optimizer (e.g., "Adam")
        stepsize: Optimizer step size
        max_iterations: Planned optimization steps
        seed: Random seed
        mapping: Fermion-to-qubit mapping identifier
        noisy: Whether the run includes noise
        depolarizing_prob: Depolarizing noise probability per wire
        amplitude_damping_prob: Amplitude damping probability per wire

    Returns:
        A dict containing all configuration details for hashing and reproducibility.
    """
    return {
        "symbols": list(symbols),
        "geometry": _round_floats(coordinates, 8),
        "basis": basis,
        "ansatz": ansatz_desc,
        "optimizer": {
            "name": optimizer_name,
            "stepsize": float(stepsize),
            "iterations_planned": int(max_iterations),
        },
        "seed": int(seed),
        "noisy": bool(noisy),
        "depolarizing_prob": float(depolarizing_prob),
        "amplitude_damping_prob": float(amplitude_damping_prob),
        "mapping": mapping.lower(),
    }


def run_signature(cfg: Dict[str, Any]) -> str:
    """Generate a stable short hash (12 hex chars) from a run configuration dictionary."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ================================================================
# FILESYSTEM UTILITIES
# ================================================================
# Base directories (package-relative)
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "package_results"
IMG_DIR = BASE_DIR / "vqe" / "images"


def ensure_dirs():
    """Ensure required directories for storing results and images exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def save_run_record(prefix: str, record: Dict[str, Any]) -> str:
    """
    Save a run record (config + results) as JSON.

    Args:
        prefix: Unique filename prefix (typically includes molecule, optimizer, and hash)
        record: Dictionary containing run configuration and results

    Returns:
        Path to the saved JSON file as a string.
    """
    ensure_dirs()
    fname = RESULTS_DIR / f"{prefix}.json"

    serializable_record = _to_serializable(record)
    with open(fname, "w") as f:
        json.dump(serializable_record, f, indent=2)

    return str(fname)
