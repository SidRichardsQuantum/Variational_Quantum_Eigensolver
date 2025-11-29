"""
vqe.io_utils
------------
Utility functions for reproducible VQE runs:

- Run configuration construction & hashing
- JSON-safe serialization
- File/directory management for results & images

This module is the single source of truth for:
- Where new package results are stored
- How run configurations are represented and hashed
"""

from __future__ import annotations

import json, os
import hashlib
from pathlib import Path
from typing import Any, Dict

# ================================================================
# BASE PATHS
# ================================================================

# Root of the repository/package (…/ <repo_root> / vqe / io_utils.py)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# New, package-scoped locations for results and images
RESULTS_DIR = BASE_DIR / "results" / "vqe"
IMG_DIR = BASE_DIR / "images" / "vqe"


# ================================================================
# INTERNAL HELPERS
# ================================================================

def _round_floats(x: Any, ndigits: int = 8) -> Any:
    """
    Recursively round floats / numpy scalars / arrays for stable hashing.

    Parameters
    ----------
    x:
        Arbitrary Python or numpy object.
    ndigits:
        Number of decimal places to round to.

    Returns
    -------
    Any
        Object of the same container structure with floats rounded.
    """
    # Plain Python float
    if isinstance(x, float):
        return round(x, ndigits)

    # Numpy scalar-like: has .item() that is a float
    try:
        if hasattr(x, "item"):
            scalar = x.item()
            if isinstance(scalar, float):
                return round(float(scalar), ndigits)
    except Exception:
        # Fall through if .item() misbehaves
        pass

    # Numpy arrays / array-like -> convert to list and recurse
    if hasattr(x, "tolist"):
        return _round_floats(x.tolist(), ndigits)

    # Python containers
    if isinstance(x, (list, tuple)):
        return type(x)(_round_floats(v, ndigits) for v in x)

    # Everything else left untouched
    return x


def _to_serializable(obj: Any) -> Any:
    """
    Convert tensors / numpy arrays / complex containers into JSON-safe types.

    Rules of thumb:
    - numpy / tensor-like with .tolist() → list / nested lists
    - numpy scalar with .item() → float (when possible)
    - dict / list / tuple → recurse
    - everything else returned unchanged
    """
    # Numpy / tensor scalar
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    # Numpy arrays and friends
    if hasattr(obj, "tolist"):
        return obj.tolist()

    # Dicts
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    # Primitive or unknown: hope json can handle it
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
    molecule_label: str | None = None,
) -> Dict[str, Any]:
    """
    Construct a canonical dictionary describing a VQE/SSVQE run configuration.

    This dictionary is used to generate a stable hash signature
    (see :func:`run_signature`) so that identical configurations
    always map to the same cache key / filename.
    """
    cfg: Dict[str, Any] = {
        "symbols": list(symbols),
        "geometry": _round_floats(coordinates, 8),
        "basis": basis,
        "ansatz": ansatz_desc,
        "optimizer": {
            "name": optimizer_name,
            "stepsize": float(stepsize),
            "iterations_planned": int(max_iterations),
        },
        "optimizer_name": optimizer_name,  # convenient flat copy
        "seed": int(seed),
        "noisy": bool(noisy),
        "depolarizing_prob": float(depolarizing_prob),
        "amplitude_damping_prob": float(amplitude_damping_prob),
        "mapping": mapping.lower(),
    }

    if molecule_label is not None:
        cfg["molecule"] = str(molecule_label)

    return cfg


def run_signature(cfg: Dict[str, Any]) -> str:
    """
    Generate a short, stable hash identifier from a run configuration.

    The configuration is JSON-serialized with sorted keys and compact
    separators, then hashed with SHA-256. The first 12 hex characters
    are used as the signature.

    Parameters
    ----------
    cfg:
        Configuration dictionary as returned by :func:`make_run_config_dict`.

    Returns
    -------
    str
        12-character hexadecimal signature.
    """
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ================================================================
# FILESYSTEM UTILITIES
# ================================================================

def ensure_dirs() -> None:
    """
    Ensure that the standard result and image directories exist.

    - RESULTS_DIR: where JSON run records are written (package_results/)
    - IMG_DIR:     where plots and figures are saved (vqe/images/)
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)


def _result_path_from_prefix(prefix: str) -> Path:
    """
    Build the full JSON path from a filename prefix (without extension).

    Parameters
    ----------
    prefix:
        Filename prefix, e.g. "H2_Adam_s0__abc123def456".

    Returns
    -------
    Path
        Path to the JSON file within RESULTS_DIR.
    """
    return RESULTS_DIR / f"{prefix}.json"


def save_run_record(prefix: str, record: Dict[str, Any]) -> str:
    """
    Save a run record (config + results) as JSON in RESULTS_DIR.

    The final path is:
        RESULTS_DIR / f"{prefix}.json"

    Parameters
    ----------
    prefix:
        Unique filename prefix, typically including molecule, optimizer,
        seed, and the run signature.
    record:
        Dictionary containing configuration, results, and any metadata.

    Returns
    -------
    str
        String path to the saved JSON file.
    """
    ensure_dirs()
    path = _result_path_from_prefix(prefix)
    serializable_record = _to_serializable(record)

    with path.open("w") as f:
        json.dump(serializable_record, f, indent=2)

    return str(path)


def make_filename_prefix(cfg: dict, *, noisy: bool, seed: int, hash_str: str, ssvqe: bool = False):
    """Return unified Option-C filename prefix."""
    # Molecule lives in config only indirectly; infer from symbols
    # OR require callers to inject molecule into cfg beforehand.
    mol = cfg.get("molecule", "MOL")  # We will fix cfg in core/ssvqe.
    
    # Ansatz string taken directly
    ans = cfg.get("ansatz", "ANSATZ")

    # Optimizer name
    if "optimizer" in cfg and "name" in cfg["optimizer"]:
        opt = cfg["optimizer"]["name"]
    else:
        opt = "OPT"

    noise_tag = "noisy" if noisy else "noiseless"
    algo_tag  = "SSVQE" if ssvqe else "VQE"

    return f"{mol}__{ans}__{opt}__{algo_tag}__{noise_tag}__s{seed}__{hash_str}"
