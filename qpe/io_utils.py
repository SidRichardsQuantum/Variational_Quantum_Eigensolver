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


def signature_hash(
    *,
    molecule: str,
    n_ancilla: int,
    t: float,
    seed: int,
    shots: Optional[int],
    noise: Optional[Dict[str, float]],
    trotter_steps: int,
) -> str:
    key = json.dumps(
        {
            "molecule": format_molecule_name(molecule),
            "n_ancilla": int(n_ancilla),
            "t": round(float(t), 10),
            "seed": int(seed),
            "trotter_steps": int(trotter_steps),
            "shots": shots,
            "noise": noise or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def cache_path(
    *,
    molecule: str,
    n_ancilla: int,
    t: float,
    seed: int,
    noise: Optional[Dict[str, float]],
    key: str,
) -> Path:
    ensure_dirs()
    mol = format_molecule_name(molecule)

    p_dep = float((noise or {}).get("p_dep", 0.0))
    p_amp = float((noise or {}).get("p_amp", 0.0))

    toks = [mol, f"{int(n_ancilla)}ancilla", f"t{int(float(t))}" if float(t).is_integer() else f"t{str(float(t)).replace('.','p')}", f"s{int(seed)}"]
    if p_dep > 0:
        toks.append(f"dep{int(round(p_dep * 100)):02d}")
    if p_amp > 0:
        toks.append(f"amp{int(round(p_amp * 100)):02d}")

    toks.append(key)
    return RESULTS_DIR / ("_".join(toks) + ".json")


def save_qpe_result(result: Dict[str, Any]) -> str:
    ensure_dirs()

    noise = result.get("noise", {}) or {}
    seed = int(result.get("seed", 0))

    key = signature_hash(
        molecule=result["molecule"],
        n_ancilla=int(result.get("n_ancilla", 0)),
        t=float(result["t"]),
        seed=seed,
        trotter_steps=int(result.get("trotter_steps", 1)),
        shots=result.get("shots", None),
        noise=noise,
    )

    path = cache_path(
        molecule=result["molecule"],
        n_ancilla=int(result.get("n_ancilla", 0)),
        t=float(result["t"]),
        seed=seed,
        noise=noise,
        key=key,
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"💾 Saved QPE result → {path}")
    return str(path)


def load_qpe_result(
    *,
    molecule: str,
    n_ancilla: int,
    t: float,
    seed: int,
    shots: Optional[int],
    noise: Optional[Dict[str, float]],
    trotter_steps: int,
) -> Optional[Dict[str, Any]]:
    key = signature_hash(
        molecule=molecule,
        n_ancilla=int(n_ancilla),
        t=float(t),
        seed=int(seed),
        trotter_steps=int(trotter_steps),
        shots=shots,
        noise=noise or {},
    )

    path = cache_path(
        molecule=molecule,
        n_ancilla=int(n_ancilla),
        t=float(t),
        seed=int(seed),
        noise=noise or {},
        key=key,
    )

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
