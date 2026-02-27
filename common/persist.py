"""
common.persist
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def round_floats(x: Any, ndigits: int = 8) -> Any:
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
        return round_floats(x.tolist(), ndigits)

    if isinstance(x, (list, tuple)):
        return type(x)(round_floats(v, ndigits) for v in x)

    if isinstance(x, dict):
        return {k: round_floats(v, ndigits) for k, v in x.items()}

    return x


def to_serializable(obj: Any) -> Any:
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
        return {k: to_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]

    return obj


def stable_hash_dict(cfg: Dict[str, Any], *, ndigits: int = 8, n_hex: int = 12) -> str:
    cfg_stable = round_floats(to_serializable(cfg), ndigits=ndigits)
    payload = json.dumps(cfg_stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n_hex]


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2, sort_keys=True)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonical_noise(
    *,
    noisy: bool,
    p_dep: float = 0.0,
    p_amp: float = 0.0,
    model: str | None = None,
) -> Dict[str, Any]:
    if not bool(noisy):
        return {}
    p_dep = float(p_dep or 0.0)
    p_amp = float(p_amp or 0.0)
    m = None if model is None else str(model).strip()
    if (p_dep == 0.0) and (p_amp == 0.0) and (m in {None, ""}):
        return {}
    return {
        "p_dep": p_dep,
        "p_amp": p_amp,
        "model": (None if (m in {None, ""}) else m),
    }


def canonical_geometry(coordinates: Any, ndigits: int = 8) -> Any:
    return round_floats(to_serializable(coordinates), ndigits=ndigits)


def stable_hash_cfg(cfg: Dict[str, Any], *, ndigits: int = 8, n_hex: int = 12) -> str:
    return stable_hash_dict(cfg, ndigits=ndigits, n_hex=n_hex)
