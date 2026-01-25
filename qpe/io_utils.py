"""
qpe.io_utils
------------
Result persistence + caching utilities for QPE.

JSON outputs:
    results/qpe/

PNG outputs:
    images/qpe/<MOLECULE>/
    (handled via common.plotting.save_plot)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from common.naming import format_molecule_name, format_token
from common.paths import results_dir
from common.persist import atomic_write_json, read_json, stable_hash_dict
from common.plotting import save_plot

RESULTS_DIR: Path = results_dir("qpe")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def signature_hash(
    *,
    molecule: str,
    n_ancilla: int,
    t: float,
    seed: int = 0,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    shots: Optional[int] = None,
    noise: Optional[Dict[str, float]] = None,
    trotter_steps: int = 1,
) -> str:
    # Canonicalise "no noise" consistently (None / {} / zeros => {})
    nz = normalize_noise(noise)

    cfg = {
        "molecule": format_molecule_name(molecule),
        "n_ancilla": int(n_ancilla),
        "t": float(t),
        "seed": int(seed),
        "trotter_steps": int(trotter_steps),
        "shots": (None if shots is None else int(shots)),
        "noise": nz,
        "mapping": str(mapping).strip().lower(),
        "unit": str(unit).strip().lower(),
    }
    return stable_hash_dict(cfg, ndigits=10, n_hex=12)


def cache_path(
    *,
    molecule: str,
    n_ancilla: int,
    t: float,
    seed: int,
    noise: Optional[Dict[str, float]],
    key: str,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
) -> Path:
    ensure_dirs()
    mol = format_molecule_name(molecule)

    p_dep = float((noise or {}).get("p_dep", 0.0))
    p_amp = float((noise or {}).get("p_amp", 0.0))

    toks = [
        mol,
        f"{int(n_ancilla)}ancilla",
        f"t{format_token(float(t))}",
        f"s{int(seed)}",
    ]
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
        mapping=result.get("mapping", "jordan_wigner"),
        unit=result.get("unit", "angstrom"),
    )

    path = cache_path(
        molecule=result["molecule"],
        n_ancilla=int(result.get("n_ancilla", 0)),
        t=float(result["t"]),
        seed=seed,
        noise=noise,
        key=key,
        mapping=result.get("mapping", "jordan_wigner"),
        unit=result.get("unit", "angstrom"),
    )

    atomic_write_json(path, result)

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
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
) -> Optional[Dict[str, Any]]:
    key = signature_hash(
        molecule=molecule,
        n_ancilla=int(n_ancilla),
        t=float(t),
        seed=int(seed),
        trotter_steps=int(trotter_steps),
        shots=shots,
        noise=noise or {},
        mapping=mapping,
        unit=unit,
    )

    path = cache_path(
        molecule=molecule,
        n_ancilla=int(n_ancilla),
        t=float(t),
        seed=int(seed),
        noise=noise or {},
        key=key,
        mapping=mapping,
        unit=unit,
    )

    if not path.exists():
        return None

    return read_json(path)


def save_qpe_plot(
    filename: str,
    *,
    molecule: str,
    show: bool = True,
) -> str:
    return save_plot(filename, kind="qpe", molecule=molecule, show=show)


def normalize_noise(noise: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not noise:
        return {}
    p_dep = float(noise.get("p_dep", 0.0))
    p_amp = float(noise.get("p_amp", 0.0))
    if (p_dep == 0.0) and (p_amp == 0.0):
        return {}
    return {"p_dep": p_dep, "p_amp": p_amp}
