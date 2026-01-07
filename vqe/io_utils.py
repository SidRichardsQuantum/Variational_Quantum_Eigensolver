"""
vqe.io_utils
------------
Reproducible VQE/SSVQE/VQD run I/O:

- Run configuration construction & hashing
- JSON-safe serialization
- File/directory management for results

Plots are handled by vqe_qpe_common.plotting.save_plot(..., molecule=...).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = BASE_DIR / "results" / "vqe"


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
    Construct a JSON-safe config dict used for hashing/caching.

    Notes
    -----
    - Callers may append extra keys (e.g. beta schedules, num_states, noise_model name).
    - We round geometry floats to stabilize hashing.
    """
    cfg: Dict[str, Any] = {
        "symbols": list(symbols),
        "geometry": _round_floats(coordinates, 8),
        "basis": str(basis),
        "ansatz": str(ansatz_desc),
        "optimizer": {
            "name": str(optimizer_name),
            "stepsize": float(stepsize),
            "iterations_planned": int(max_iterations),
        },
        "optimizer_name": str(optimizer_name),
        "seed": int(seed),
        "noisy": bool(noisy),
        "depolarizing_prob": float(depolarizing_prob),
        "amplitude_damping_prob": float(amplitude_damping_prob),
        "mapping": str(mapping).lower(),
    }

    if molecule_label is not None:
        cfg["molecule"] = str(molecule_label)

    return cfg


def run_signature(cfg: Dict[str, Any]) -> str:
    """
    Stable short hash used to identify a run config.

    Important: cfg should already be JSON-safe (or at least JSON-dumpable).
    """
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _result_path_from_prefix(prefix: str) -> Path:
    return RESULTS_DIR / f"{prefix}.json"


def save_run_record(prefix: str, record: Dict[str, Any]) -> str:
    """Save run record JSON under results/vqe/<prefix>.json."""
    ensure_dirs()
    path = _result_path_from_prefix(prefix)
    serializable_record = _to_serializable(record)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_record, f, indent=2)
    return str(path)


def make_filename_prefix(
    cfg: dict,
    *,
    noisy: bool,
    seed: int,
    hash_str: str,
    algo: Optional[str] = None,
    ssvqe: Optional[bool] = None,
) -> str:
    """
    Build a human-readable filename prefix.

    Preferred usage (new):
        make_filename_prefix(cfg, noisy=noisy, seed=seed, hash_str=sig, algo="VQD")

    Backwards compatible usage (legacy):
        make_filename_prefix(..., ssvqe=True/False)

    Parameters
    ----------
    algo
        One of {"VQE","SSVQE","VQD"} (case-insensitive). If provided, it wins.
    ssvqe
        Legacy flag: True -> "SSVQE", False -> "VQE". Ignored if algo is provided.
    """
    mol = cfg.get("molecule", "MOL")
    ans = cfg.get("ansatz", "ANSATZ")

    opt = "OPT"
    if isinstance(cfg.get("optimizer"), dict) and "name" in cfg["optimizer"]:
        opt = cfg["optimizer"]["name"]

    noise_tag = "noisy" if noisy else "noiseless"

    if algo is not None:
        algo_tag = str(algo).strip().upper()
        if algo_tag not in {"VQE", "SSVQE", "VQD"}:
            raise ValueError("algo must be one of: 'VQE', 'SSVQE', 'VQD'")
    else:
        # legacy behavior
        algo_tag = "SSVQE" if bool(ssvqe) else "VQE"

    return f"{mol}__{ans}__{opt}__{algo_tag}__{noise_tag}__s{int(seed)}__{hash_str}"
