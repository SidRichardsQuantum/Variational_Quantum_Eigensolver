"""
vqe.io_utils
------------
Reproducible VQE/SSVQE run I/O:

- Run configuration construction & hashing
- JSON-safe serialization
- File/directory management for results

Plots are handled by vqe_qpe_common.plotting.save_plot(..., molecule=...).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = BASE_DIR / "results" / "vqe"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _round_floats(x: Any, ndigits: int = 8) -> Any:
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

    return x


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist"):
        return obj.tolist()

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
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _result_path_from_prefix(prefix: str) -> Path:
    return RESULTS_DIR / f"{prefix}.json"


def save_run_record(prefix: str, record: Dict[str, Any]) -> str:
    ensure_dirs()
    path = _result_path_from_prefix(prefix)
    serializable_record = _to_serializable(record)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable_record, f, indent=2)
    return str(path)


def make_filename_prefix(
    cfg: dict, *, noisy: bool, seed: int, hash_str: str, ssvqe: bool = False
) -> str:
    mol = cfg.get("molecule", "MOL")
    ans = cfg.get("ansatz", "ANSATZ")

    opt = "OPT"
    if isinstance(cfg.get("optimizer"), dict) and "name" in cfg["optimizer"]:
        opt = cfg["optimizer"]["name"]

    noise_tag = "noisy" if noisy else "noiseless"
    algo_tag = "SSVQE" if ssvqe else "VQE"

    return f"{mol}__{ans}__{opt}__{algo_tag}__{noise_tag}__s{int(seed)}__{hash_str}"
