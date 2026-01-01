"""
qpe.io_utils
------------
Result persistence + caching utilities for QPE.

JSON outputs:
    results/qpe/

PNG outputs:
    images/qpe/<MOLECULE>/
    (handled via vqe_qpe_common.plotting.save_plot)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from vqe_qpe_common.plotting import format_molecule_name, save_plot

BASE_DIR: Path = Path(__file__).resolve().parent.parent
RESULTS_DIR: Path = BASE_DIR / "results" / "qpe"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Backwards-compatible alias (optional, but harmless)
def ensure_qpe_dirs() -> None:
    ensure_dirs()


def signature_hash(
    *,
    molecule: str,
    n_ancilla: int,
    t: float,
    shots: Optional[int],
    noise: Optional[Dict[str, float]],
    trotter_steps: int,
) -> str:
    key = json.dumps(
        {
            "molecule": format_molecule_name(molecule),
            "n_ancilla": int(n_ancilla),
            "t": round(float(t), 10),
            "trotter_steps": int(trotter_steps),
            "shots": shots,
            "noise": noise or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def cache_path(molecule: str, key: str) -> Path:
    ensure_dirs()
    mol = format_molecule_name(molecule)
    return RESULTS_DIR / f"{mol}__QPE__{key}.json"


def save_qpe_result(result: Dict[str, Any]) -> str:
    ensure_dirs()

    key = signature_hash(
        molecule=result["molecule"],
        n_ancilla=int(result.get("n_ancilla", result.get("n_ancilla", 0))),
        t=float(result["t"]),
        trotter_steps=int(result.get("trotter_steps", 1)),
        shots=result.get("shots", None),
        noise=result.get("noise", {}) or {},
    )

    path = cache_path(result["molecule"], key)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"💾 Saved QPE result → {path}")
    return str(path)


def load_qpe_result(molecule: str, key: str) -> Optional[Dict[str, Any]]:
    path = cache_path(molecule, key)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_qpe_plot(
    filename: str,
    *,
    molecule: str,
    show: bool = True,
) -> str:
    return save_plot(filename, kind="qpe", molecule=molecule, show=show)
