import os, json, glob, hashlib
from typing import Any, Dict

def _round_floats(x: Any, ndigits: int = 8):
    """Recursively round floats / arrays / lists for stable hashing."""
    if isinstance(x, float):
        return round(x, ndigits)
    try:
        if hasattr(x, "item") and isinstance(x.item(), float):
            return round(float(x), ndigits)
    except Exception:
        pass
    if hasattr(x, "tolist"):
        return _round_floats(x.tolist(), ndigits)
    if isinstance(x, (list, tuple)):
        return type(x)(_round_floats(v, ndigits) for v in x)
    return x


def make_run_config_dict(symbols, coordinates, basis, ansatz_desc, optimizer_name,
                         stepsize, max_iterations, seed, noisy=False,
                         depolarizing_prob=0.0, amplitude_damping_prob=0.0) -> Dict[str, Any]:
    """Canonical config for hashing & reproducibility."""
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
        "noisy": noisy,
        "depolarizing_prob": float(depolarizing_prob),
        "amplitude_damping_prob": float(amplitude_damping_prob),
    }


def run_signature(cfg: Dict[str, Any]) -> str:
    """Stable short hash of the config (12 hex chars)."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# --- directories ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

def _to_serializable(obj: Any):
    """Convert tensors/numpy objects recursively into Python types for JSON."""
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

def save_run_record(prefix: str, record: Dict[str, Any]) -> str:
    """Save a run record as JSON, converting non-serializable types."""
    ensure_dirs()
    fname = os.path.join(RESULTS_DIR, f"{prefix}.json")

    # Convert tensors and numpy arrays to native Python types
    serializable_record = _to_serializable(record)

    with open(fname, "w") as f:
        json.dump(serializable_record, f, indent=2)
    return fname
