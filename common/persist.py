"""
common.persist
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pennylane as qml


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
        return [round_floats(v, ndigits) for v in x]

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


def cached_compute_runtime(result: Dict[str, Any]) -> float | None:
    """
    Return the stored compute runtime from a cached result when available.

    Older cache records may predate runtime metadata entirely. In that case,
    return None so callers can treat the cache entry as stale and recompute.
    """
    val = result.get("compute_runtime_s", result.get("runtime_s"))
    if val is None:
        return None
    return float(val)


def canonical_noise(
    *,
    noisy: bool,
    p_dep: float = 0.0,
    p_amp: float = 0.0,
    p_phase_damp: float = 0.0,
    p_bit_flip: float = 0.0,
    p_phase_flip: float = 0.0,
    model: str | None = None,
) -> Dict[str, Any]:
    if not bool(noisy):
        return {}
    vals = {
        "p_dep": float(p_dep or 0.0),
        "p_amp": float(p_amp or 0.0),
        "p_phase_damp": float(p_phase_damp or 0.0),
        "p_bit_flip": float(p_bit_flip or 0.0),
        "p_phase_flip": float(p_phase_flip or 0.0),
    }
    m = None if model is None else str(model).strip()
    out: Dict[str, Any] = {k: v for k, v in vals.items() if v != 0.0}
    if (not out) and (m in {None, ""}):
        return {}
    if m not in {None, ""}:
        out["model"] = m
    return out


def canonical_geometry(coordinates: Any, ndigits: int = 8) -> Any:
    return round_floats(to_serializable(coordinates), ndigits=ndigits)


def canonical_hamiltonian(hamiltonian: Any, ndigits: int = 10) -> Dict[str, Any]:
    """
    Return a stable JSON-safe representation of a qubit Hamiltonian.

    The representation is based on PennyLane's PauliSentence so equivalent
    Pauli-term orderings produce the same cache key.
    """
    sentence = qml.pauli.pauli_sentence(hamiltonian)
    terms = []
    for word, coeff in sentence.items():
        coeff_c = complex(coeff)
        if abs(coeff_c) <= 1e-12:
            continue
        ops = [[int(wire), str(axis)] for wire, axis in sorted(word.items())]
        terms.append(
            {
                "ops": ops,
                "coeff_real": round(float(coeff_c.real), ndigits),
                "coeff_imag": round(float(coeff_c.imag), ndigits),
            }
        )

    terms.sort(
        key=lambda term: (
            term["ops"],
            term["coeff_real"],
            term["coeff_imag"],
        )
    )
    return {"pauli_terms": terms}


def stable_hash_cfg(cfg: Dict[str, Any], *, ndigits: int = 8, n_hex: int = 12) -> str:
    return stable_hash_dict(cfg, ndigits=ndigits, n_hex=n_hex)
