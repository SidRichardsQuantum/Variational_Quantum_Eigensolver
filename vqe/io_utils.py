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

from common.paths import results_dir
from common.persist import (
    atomic_write_json,
    canonical_geometry,
    canonical_noise,
    read_json,
    stable_hash_cfg,
)

RESULTS_DIR: Path = results_dir("vqe")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
        model=None,
    )

    cfg: Dict[str, Any] = {
        "molecule": (None if molecule_label is None else str(molecule_label)),
        "symbols": list(symbols),
        "geometry": canonical_geometry(coordinates, ndigits=8),
        "basis": str(basis).strip().lower(),
        "mapping": str(mapping).strip().lower(),
        "seed": int(seed),
        "noisy": bool(bool(noise)),
        "noise": noise,
        "ansatz": str(ansatz_desc),
        "optimizer": {
            "name": str(optimizer_name),
            "stepsize": float(stepsize),
            "iterations_planned": int(max_iterations),
        },
    }

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
    """Save run record JSON under results/vqe/<prefix>.json."""
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
    algo: Optional[str] = None,
) -> str:
    from common.naming import format_molecule_name
    from common.plotting import slug_token

    mol = str(cfg.get("molecule") or "MOL").strip()
    ans = str(cfg.get("ansatz", "ANSATZ")).strip()

    opt = "OPT"
    if isinstance(cfg.get("optimizer"), dict) and "name" in cfg["optimizer"]:
        opt = str(cfg["optimizer"]["name"]).strip()

    def _noise_tokens(noise: dict) -> list[str]:
        toks: list[str] = []
        p_dep = float((noise or {}).get("p_dep", 0.0))
        p_amp = float((noise or {}).get("p_amp", 0.0))

        def _pct(p: float) -> str:
            return f"{int(round(p * 100)):02d}"

        if p_dep > 0.0:
            toks.append(f"dep{_pct(p_dep)}")
        if p_amp > 0.0:
            toks.append(f"amp{_pct(p_amp)}")
        return toks

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

    parts: list[str] = [
        format_molecule_name(mol),
        slug_token(ans),
        slug_token(opt),
    ]

    if algo_tok is not None:
        parts.append(algo_tok)

    noise = cfg.get("noise", {}) or {}
    parts.append("noisy" if bool(noise) else "noiseless")
    parts.extend(_noise_tokens(noise))

    parts.append(f"s{int(seed)}")
    parts.append(str(hash_str).strip())

    return "_".join(parts)
