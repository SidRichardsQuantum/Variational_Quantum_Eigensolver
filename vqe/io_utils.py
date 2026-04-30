"""
vqe.io_utils
------------
Reproducible VQE/SSVQE/VQD run I/O:

- Run configuration construction & hashing
- JSON-safe serialization
- File/directory management for results

Plots are handled by common.plotting.save_plot(..., molecule=...).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from common.environment import ensure_environment_metadata
from common.paths import results_dir
from common.persist import (
    atomic_write_json,
    canonical_geometry,
    canonical_noise,
    read_json,
    stable_hash_cfg,
)
from vqe.ansatz import canonicalize_ansatz_name
from vqe.optimizer import canonicalize_optimizer_name

RESULTS_DIR: Path = results_dir("vqe")


def ensure_dirs() -> None:
    """Create the VQE results directory if it does not already exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _json_safe_mapping(values: dict[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}

    def convert(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in sorted(value.items())}
        if hasattr(value, "tolist"):
            return convert(value.tolist())
        return str(value)

    return {str(k): convert(v) for k, v in sorted(values.items())}


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
    phase_damping_prob: float = 0.0,
    bit_flip_prob: float = 0.0,
    phase_flip_prob: float = 0.0,
    molecule_label: str | None = None,
    charge: int = 0,
    unit: str = "angstrom",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    ansatz_kwargs: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Construct a JSON-safe config dict used for hashing/caching.

    Notes
    -----
    - Callers may append extra keys (e.g. beta schedules, num_states, noise_model name).
    - We round geometry floats to stabilize hashing.
    """
    noise = canonical_noise(
        noisy=bool(noisy),
        p_dep=float(depolarizing_prob),
        p_amp=float(amplitude_damping_prob),
        p_phase_damp=float(phase_damping_prob),
        p_bit_flip=float(bit_flip_prob),
        p_phase_flip=float(phase_flip_prob),
        model=None,
    )

    cfg: Dict[str, Any] = {
        "molecule": (None if molecule_label is None else str(molecule_label)),
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
        "ansatz": canonicalize_ansatz_name(ansatz_desc),
        "ansatz_kwargs": _json_safe_mapping(ansatz_kwargs),
        "optimizer": {
            "name": canonicalize_optimizer_name(optimizer_name),
            "stepsize": float(stepsize),
            "iterations_planned": int(max_iterations),
        },
    }

    return cfg


def run_signature(cfg: Dict[str, Any]) -> str:
    """Return the stable hash used in VQE-family cache filenames."""
    return stable_hash_cfg(cfg, ndigits=8, n_hex=12)


def load_run_record(prefix: str) -> Dict[str, Any] | None:
    path = _result_path_from_prefix(prefix)
    if not path.exists():
        return None
    record = read_json(path)
    if isinstance(record.get("result"), dict):
        ensure_environment_metadata(record["result"])
    return record


def _result_path_from_prefix(prefix: str) -> Path:
    return RESULTS_DIR / f"{prefix}.json"


def save_run_record(prefix: str, record: Dict[str, Any]) -> str:
    """Save run record JSON under results/vqe/<prefix>.json."""
    ensure_dirs()
    if isinstance(record.get("result"), dict):
        ensure_environment_metadata(record["result"])
    path = _result_path_from_prefix(prefix)
    atomic_write_json(path, record)
    return str(path)


def make_filename_prefix(
    cfg: dict,
    *,
    noisy: bool,
    seed: int,
    hash_str: str,
    algo: Optional[str] = None,
) -> str:
    from common.naming import format_molecule_name
    from common.plotting import build_filename, slug_token

    mol = str(cfg.get("molecule") or "MOL").strip()
    ans = str(cfg.get("ansatz", "ANSATZ")).strip()

    opt = "OPT"
    if isinstance(cfg.get("optimizer"), dict) and "name" in cfg["optimizer"]:
        opt = str(cfg["optimizer"]["name"]).strip()

    algo_tok: Optional[str] = None
    if algo is not None:
        a = str(algo).strip().lower()
        if a not in {
            "vqe",
            "ssvqe",
            "vqd",
            "qse",
            "lr",
            "eom_vqe",
            "eom_qse",
        }:
            raise ValueError(
                "algo must be one of: "
                "'vqe', 'ssvqe', 'vqd', 'qse', 'lr', 'eom_vqe', 'eom_qse'"
            )
        if a in {"ssvqe", "vqd", "qse", "lr", "eom_vqe", "eom_qse"}:
            algo_tok = a

    noise = cfg.get("noise", {}) or {}
    p_dep = float((noise or {}).get("p_dep", 0.0))
    p_amp = float((noise or {}).get("p_amp", 0.0))
    p_phase = float((noise or {}).get("p_phase_damp", 0.0))
    p_bit = float((noise or {}).get("p_bit_flip", 0.0))
    p_phase_flip = float((noise or {}).get("p_phase_flip", 0.0))

    parts: list[str] = [
        format_molecule_name(mol),
        slug_token(ans),
        slug_token(opt),
    ]

    if algo_tok is not None:
        parts.append(algo_tok)

    parts.append(
        "noisy"
        if (
            p_dep > 0.0
            or p_amp > 0.0
            or p_phase > 0.0
            or p_bit > 0.0
            or p_phase_flip > 0.0
        )
        else "noiseless"
    )

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
        parts.append(f"phase{slug_token(p_phase)}")
    if p_bit > 0.0:
        parts.append(f"bit{slug_token(p_bit)}")
    if p_phase_flip > 0.0:
        parts.append(f"phaseflip{slug_token(p_phase_flip)}")

    parts.append(f"s{int(seed)}")
    parts.append(str(hash_str).strip())

    return "_".join([p for p in parts if str(p).strip() != ""])
