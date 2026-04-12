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

from pathlib import Path
from typing import Any, Dict

from common.paths import results_dir
from common.persist import (
    atomic_write_json,
    canonical_geometry,
    canonical_noise,
    read_json,
    stable_hash_cfg,
)

RESULTS_DIR: Path = results_dir("qite")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_run_config_dict(
    *,
    symbols,
    coordinates,
    basis,
    charge: int,
    unit: str,
    seed: int,
    mapping: str,
    noisy: bool,
    depolarizing_prob: float,
    amplitude_damping_prob: float,
    phase_damping_prob: float,
    bit_flip_prob: float,
    phase_flip_prob: float,
    dtau: float,
    steps: int,
    molecule_label: str,
    ansatz_name: str,
    noise_model_name: str | None = None,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    fd_eps: float | None = None,
    reg: float | None = None,
    solver: str | None = None,
    pinv_rcond: float | None = None,
):
    """
    Build a stable, JSON-serialisable config dict for caching.

    Notes
    -----
    - VarQITE numerics are included when provided so they participate in caching.
    """
    noise = canonical_noise(
        noisy=bool(noisy),
        p_dep=float(depolarizing_prob),
        p_amp=float(amplitude_damping_prob),
        p_phase_damp=float(phase_damping_prob),
        p_bit_flip=float(bit_flip_prob),
        p_phase_flip=float(phase_flip_prob),
        model=noise_model_name,
    )

    cfg = {
        "molecule": str(molecule_label),
        "symbols": list(symbols),
        "geometry": canonical_geometry(coordinates, ndigits=8),
        "basis": str(basis).strip().lower(),
        "charge": int(charge),
        "unit": str(unit).strip().lower(),
        "mapping": str(mapping).strip().lower(),
        "active_electrons": (
            None if active_electrons is None else int(active_electrons)
        ),
        "active_orbitals": (None if active_orbitals is None else int(active_orbitals)),
        "seed": int(seed),
        "noisy": bool(bool(noise)),
        "noise": noise,
        "dtau": float(dtau),
        "steps": int(steps),
        "ansatz": str(ansatz_name),
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
    return stable_hash_cfg(cfg, ndigits=8, n_hex=12)


def load_run_record(prefix: str) -> Dict[str, Any] | None:
    path = _result_path_from_prefix(prefix)
    if not path.exists():
        return None
    return read_json(path)


def _result_path_from_prefix(prefix: str) -> Path:
    return RESULTS_DIR / f"{prefix}.json"


def save_run_record(prefix: str, record: Dict[str, Any]) -> str:
    """Save run record JSON under results/qite/<prefix>.json."""
    ensure_dirs()
    path = _result_path_from_prefix(prefix)
    atomic_write_json(path, record)
    return str(path)


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
    from common.naming import format_molecule_name, format_token
    from common.plotting import slug_token

    mol = str(cfg.get("molecule", cfg.get("molecule_label", "UNK")))

    # Normalise molecule label for filesystem
    mol_fs = format_molecule_name(mol)

    ansatz = str(cfg.get("ansatz", ""))
    mapping = str(cfg.get("mapping", ""))
    steps = int(cfg.get("steps", 0))
    dtau = float(cfg.get("dtau", 0.0))

    algo_tag = str(algo).strip().lower() if algo else "qite"
    dtau_tok = format_token(dtau)

    noise = cfg.get("noise", {}) or {}
    p_dep = float(noise.get("p_dep", 0.0))
    p_amp = float(noise.get("p_amp", 0.0))
    p_phase = float(noise.get("p_phase_damp", 0.0))
    p_bit = float(noise.get("p_bit_flip", 0.0))
    p_phase_flip = float(noise.get("p_phase_flip", 0.0))

    parts: list[str] = [
        algo_tag,
        mol_fs,
        slug_token(ansatz),
        slug_token(mapping),
        "noisy" if bool(noise) else "noiseless",
        f"steps{steps}",
        f"dtau{dtau_tok}",
    ]
    from common.plotting import build_filename

    if p_dep > 0.0 or p_amp > 0.0:
        noise_png = build_filename(
            topic="x",
            dep=(p_dep if p_dep > 0.0 else None),
            amp=(p_amp if p_amp > 0.0 else None),
            noise_scan=False,
            multi_seed=False,
        )
        noise_mid = noise_png.removesuffix(".png")
        if noise_mid != "x":
            parts.append(noise_mid)
    if p_phase > 0.0:
        parts.append(f"phase{format_token(p_phase)}")
    if p_bit > 0.0:
        parts.append(f"bit{format_token(p_bit)}")
    if p_phase_flip > 0.0:
        parts.append(f"phaseflip{format_token(p_phase_flip)}")

    parts.append(f"s{int(seed)}")
    parts.append(str(hash_str).strip())
    return "_".join(parts)
