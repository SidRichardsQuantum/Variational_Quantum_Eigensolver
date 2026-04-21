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

from common.naming import format_molecule_name
from common.paths import results_dir
from common.persist import (
    atomic_write_json,
    canonical_geometry,
    canonical_noise,
    read_json,
    stable_hash_cfg,
)
from common.plotting import build_filename, save_plot

RESULTS_DIR: Path = results_dir("qpe")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def signature_hash(
    *,
    molecule: str,
    symbols: list[str],
    geometry,
    basis: str,
    charge: int,
    n_ancilla: int,
    t: float,
    seed: int = 0,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    hamiltonian: Dict[str, Any] | None = None,
    reference_state: list[int] | None = None,
    shots: Optional[int] = None,
    noise: Optional[Dict[str, float]] = None,
    trotter_steps: int = 1,
) -> str:
    """
    Return the stable cache key for a QPE run configuration.

    The key includes normalized geometry, basis, charge, mapping, unit,
    active-space settings, QPE controls, shot mode, noise parameters, and expert
    Hamiltonian/reference fingerprints when provided.
    """
    nz = canonical_noise(
        noisy=True,
        p_dep=float((noise or {}).get("p_dep", 0.0)),
        p_amp=float((noise or {}).get("p_amp", 0.0)),
        p_phase_damp=float((noise or {}).get("p_phase_damp", 0.0)),
        p_bit_flip=float((noise or {}).get("p_bit_flip", 0.0)),
        p_phase_flip=float((noise or {}).get("p_phase_flip", 0.0)),
        model=None,
    )

    cfg = {
        "molecule": format_molecule_name(molecule),
        "symbols": list(symbols),
        "geometry": canonical_geometry(geometry, ndigits=8),
        "basis": str(basis).strip().lower(),
        "charge": int(charge),
        "n_ancilla": int(n_ancilla),
        "t": float(t),
        "seed": int(seed),
        "trotter_steps": int(trotter_steps),
        "shots": (None if shots is None else int(shots)),
        "noise": nz,
        "mapping": str(mapping).strip().lower(),
        "unit": str(unit).strip().lower(),
        "active_electrons": (
            None if active_electrons is None else int(active_electrons)
        ),
        "active_orbitals": (None if active_orbitals is None else int(active_orbitals)),
        "hamiltonian": hamiltonian,
        "reference_state": reference_state,
    }
    return stable_hash_cfg(cfg, ndigits=10, n_hex=12)


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

    nz = canonical_noise(
        noisy=True,
        p_dep=float((noise or {}).get("p_dep", 0.0)),
        p_amp=float((noise or {}).get("p_amp", 0.0)),
        p_phase_damp=float((noise or {}).get("p_phase_damp", 0.0)),
        p_bit_flip=float((noise or {}).get("p_bit_flip", 0.0)),
        p_phase_flip=float((noise or {}).get("p_phase_flip", 0.0)),
        model=None,
    )
    p_dep = float(nz.get("p_dep", 0.0))
    p_amp = float(nz.get("p_amp", 0.0))
    p_phase = float(nz.get("p_phase_damp", 0.0))
    p_bit = float(nz.get("p_bit_flip", 0.0))
    p_phase_flip = float(nz.get("p_phase_flip", 0.0))

    fname = build_filename(
        topic="qpe",
        ancilla=int(n_ancilla),
        t=float(t),
        dep=(p_dep if p_dep > 0.0 else None),
        amp=(p_amp if p_amp > 0.0 else None),
        noise_scan=False,
        multi_seed=False,
        seed=int(seed),
        tag=str(key).strip(),
    )
    extra_noise = []
    if p_phase > 0.0:
        extra_noise.append(f"phase{p_phase:g}")
    if p_bit > 0.0:
        extra_noise.append(f"bit{p_bit:g}")
    if p_phase_flip > 0.0:
        extra_noise.append(f"phaseflip{p_phase_flip:g}")
    if extra_noise:
        fname = fname.removesuffix(".png") + "_" + "_".join(extra_noise) + ".png"

    fname = fname.removesuffix(".png") + ".json"
    return RESULTS_DIR / f"{mol}_{fname}"


def save_qpe_result(result: Dict[str, Any]) -> str:
    """
    Persist a QPE result as a cache JSON file.

    The destination path is derived from the same normalized signature used by
    ``load_qpe_result``. The input dictionary must contain the run metadata
    emitted by ``run_qpe``.
    """
    ensure_dirs()

    noise = result.get("noise", {}) or {}
    seed = int(result.get("seed", 0))

    key = signature_hash(
        molecule=result["molecule"],
        symbols=result["symbols"],
        geometry=result["geometry"],
        basis=result["basis"],
        charge=int(result["charge"]),
        n_ancilla=int(result.get("n_ancilla", 0)),
        t=float(result["t"]),
        seed=seed,
        trotter_steps=int(result.get("trotter_steps", 1)),
        shots=result.get("shots", None),
        noise=noise,
        mapping=result.get("mapping", "jordan_wigner"),
        unit=result.get("unit", "angstrom"),
        active_electrons=result.get("active_electrons"),
        active_orbitals=result.get("active_orbitals"),
        hamiltonian=result.get("hamiltonian"),
        reference_state=(
            result.get("hf_state") if result.get("hamiltonian") is not None else None
        ),
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
    return str(path)


def load_qpe_result(
    *,
    molecule: str,
    symbols: list[str],
    geometry,
    basis: str,
    charge: int,
    n_ancilla: int,
    t: float,
    seed: int,
    shots: Optional[int],
    noise: Optional[Dict[str, float]],
    trotter_steps: int,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    hamiltonian: Dict[str, Any] | None = None,
    reference_state: list[int] | None = None,
) -> Optional[Dict[str, Any]]:
    """
    Load a cached QPE result matching the provided normalized configuration.

    Returns ``None`` when no matching cache file exists. Callers are responsible
    for validating whether old cache records contain any runtime metadata they
    require.
    """
    key = signature_hash(
        molecule=molecule,
        symbols=symbols,
        geometry=geometry,
        basis=basis,
        charge=int(charge),
        n_ancilla=int(n_ancilla),
        t=float(t),
        seed=int(seed),
        trotter_steps=int(trotter_steps),
        shots=shots,
        noise=noise or {},
        mapping=mapping,
        unit=unit,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian=hamiltonian,
        reference_state=reference_state,
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
    return canonical_noise(
        noisy=True,
        p_dep=float(noise.get("p_dep", 0.0)),
        p_amp=float(noise.get("p_amp", 0.0)),
        p_phase_damp=float(noise.get("p_phase_damp", 0.0)),
        p_bit_flip=float(noise.get("p_bit_flip", 0.0)),
        p_phase_flip=float(noise.get("p_phase_flip", 0.0)),
        model=None,
    )
