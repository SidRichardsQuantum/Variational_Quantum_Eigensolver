"""
qite.io_utils
-------------
Reproducible QITE run I/O:

- Run configuration construction & hashing
- JSON-safe serialization
- File/directory management for results

Plots are handled by qite.visualize (images/qite/<MOLECULE>/...).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = BASE_DIR / "results" / "qite"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _round_floats(x: Any, ndigits: int = 8) -> Any:
    """Round floats recursively to stabilize config hashing against tiny fp noise."""
    if isinstance(x, float):
        return round(x, ndigits)

    try:
        if hasattr(x, "item"):
            scalar = x.item()
            if isinstance(scalar, float):
                return round(float(scalar), ndigits)
    except Exception:
        pass

    if hasattr(x, "tolist"):
        return _round_floats(x.tolist(), ndigits)

    if isinstance(x, (list, tuple)):
        return type(x)(_round_floats(v, ndigits) for v in x)

    if isinstance(x, dict):
        return {k: _round_floats(v, ndigits) for k, v in x.items()}

    return x


def _to_serializable(obj: Any) -> Any:
    """Convert nested objects (numpy / pennylane types) to JSON-serializable types."""
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    return obj


def make_run_config_dict(
    *,
    symbols,
    coordinates,
    basis,
    seed: int,
    mapping: str,
    noisy: bool,
    depolarizing_prob: float,
    amplitude_damping_prob: float,
    dtau: float,
    steps: int,
    molecule_label: str,
    ansatz_name: str | None = None,
    ansatz: str | None = None,
    ansatz_desc: str | None = None,
    noise_model_name: str | None = None,
    # VarQITE numerics (optional; included if provided)
    fd_eps: float | None = None,
    reg: float | None = None,
    solver: str | None = None,
    pinv_rcond: float | None = None,
    **_ignored,
):
    """
    Build a stable, JSON-serialisable config dict for caching.

    Notes
    -----
    - We accept several aliases (ansatz_name/ansatz/ansatz_desc).
    - We ignore unknown fields so qite.core can evolve without breaking caching.
    - VarQITE numerics are included when provided so they participate in caching.
    """
    # Resolve ansatz label with a simple priority order
    ansatz_label = (
        ansatz_name
        if ansatz_name is not None
        else (
            ansatz
            if ansatz is not None
            else (ansatz_desc if ansatz_desc is not None else "")
        )
    )

    cfg = {
        "molecule": str(molecule_label),
        "symbols": list(symbols),
        "coordinates": coordinates.tolist(),
        "basis": str(basis),
        "seed": int(seed),
        "mapping": str(mapping),
        "noisy": bool(noisy),
        "depolarizing_prob": float(depolarizing_prob),
        "amplitude_damping_prob": float(amplitude_damping_prob),
        "noise_model_name": (
            None if noise_model_name is None else str(noise_model_name)
        ),
        "dtau": float(dtau),
        "steps": int(steps),
        "ansatz": str(ansatz_label),
    }

    # Optional VarQITE numerics (kept explicit for stable cache keys)
    if fd_eps is not None:
        cfg["fd_eps"] = float(fd_eps)
    if reg is not None:
        cfg["reg"] = float(reg)
    if solver is not None:
        cfg["solver"] = str(solver)
    if pinv_rcond is not None:
        cfg["pinv_rcond"] = float(pinv_rcond)

    return cfg


def run_signature(cfg: Dict[str, Any]) -> str:
    """
    Stable short hash used to identify a run config.

    Important: cfg should already be JSON-safe (or at least JSON-dumpable).
    We additionally round floats recursively to reduce accidental cache misses.
    """
    cfg_stable = _round_floats(_to_serializable(cfg))
    payload = json.dumps(cfg_stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _result_path_from_prefix(prefix: str) -> Path:
    return RESULTS_DIR / f"{prefix}.json"


def save_run_record(prefix: str, record: Dict[str, Any]) -> str:
    """Save run record JSON under results/qite/<prefix>.json."""
    ensure_dirs()
    path = _result_path_from_prefix(prefix)
    serializable_record = _to_serializable(record)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_record, f, indent=2)
    return str(path)


def is_effectively_noisy(
    noisy: bool,
    depolarizing_prob: float,
    amplitude_damping_prob: float,
    noise_model=None,
) -> bool:
    if not bool(noisy):
        return False
    return (
        float(depolarizing_prob) != 0.0
        or float(amplitude_damping_prob) != 0.0
        or (noise_model is not None)
    )


def make_filename_prefix(
    cfg: dict,
    *,
    noisy: bool,
    seed: int,
    hash_str: str,
    algo: str | None = None,
    **_ignored,
) -> str:
    """
    Build a stable filename prefix for results/images.

    Notes
    -----
    Accepts extra kwargs so qite.core can evolve without breaking I/O.
    """
    mol = str(cfg.get("molecule", cfg.get("molecule_label", "UNK")))

    # Normalise molecule label for filesystem
    mol_fs = mol.replace("+", "plus").replace(" ", "_").replace("/", "_")

    ansatz = str(cfg.get("ansatz", ""))
    mapping = str(cfg.get("mapping", ""))
    steps = int(cfg.get("steps", 0))
    dtau = float(cfg.get("dtau", 0.0))

    algo_tag = str(algo).strip().lower() if algo else "qite"
    noise_tag = "noisy" if bool(noisy) else "noiseless"

    return (
        f"{algo_tag}__{mol_fs}__{ansatz}__{mapping}__"
        f"{noise_tag}__steps{steps}__dtau{dtau:g}__s{int(seed)}__{hash_str}"
    )
