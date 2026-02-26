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
    read_json,
    round_floats,
    stable_hash_dict,
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
    # Canonicalise noise: if not noisy, do not allow nonzero noise params to pollute cache keys / filenames.
    if not bool(noisy):
        depolarizing_prob = 0.0
        amplitude_damping_prob = 0.0

    cfg: Dict[str, Any] = {
        "symbols": list(symbols),
        "geometry": round_floats(coordinates, 8),
        "basis": str(basis).lower(),
        "ansatz": str(ansatz_desc),
        "optimizer": {
            "name": str(optimizer_name),
            "stepsize": float(stepsize),
            "iterations_planned": int(max_iterations),
        },
        "optimizer_name": str(optimizer_name),
        "seed": int(seed),
        "noisy": bool(noisy),
        "depolarizing_prob": float(depolarizing_prob),
        "amplitude_damping_prob": float(amplitude_damping_prob),
        "mapping": str(mapping).lower(),
    }

    if molecule_label is not None:
        cfg["molecule"] = str(molecule_label)

    return cfg


def run_signature(cfg: Dict[str, Any]) -> str:
    return stable_hash_dict(cfg, ndigits=8, n_hex=12)


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


def is_effectively_noisy(
    noisy: bool,
    depolarizing_prob: float,
    amplitude_damping_prob: float,
    noise_model=None,
) -> bool:
    if not bool(noisy):
        return False
    return (
        float(depolarizing_prob) != 0.0
        or float(amplitude_damping_prob) != 0.0
        or (noise_model is not None)
    )


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

    mol = str(cfg.get("molecule", "MOL")).strip()
    ans = str(cfg.get("ansatz", "ANSATZ")).strip()

    opt = "OPT"
    if isinstance(cfg.get("optimizer"), dict) and "name" in cfg["optimizer"]:
        opt = str(cfg["optimizer"]["name"]).strip()

    def _slug_general(x: str) -> str:
        s = str(x).strip().lower()
        s = s.replace("+", "plus")
        s = s.replace(" ", "_")
        s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_") or "x"

    def _slug_molecule_label(mol: str) -> str:
        s = str(mol).strip()
        s = s.replace("+", "plus")
        s = s.replace(" ", "_")
        s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_") or "MOL"

    def _noise_tokens(dep: float, amp: float) -> list[str]:
        toks: list[str] = []
        dep_f = float(dep or 0.0)
        amp_f = float(amp or 0.0)

        def _pct(p: float) -> str:
            return f"{int(round(p * 100)):02d}"

        if dep_f > 0.0:
            toks.append(f"dep{_pct(dep_f)}")
        if amp_f > 0.0:
            toks.append(f"amp{_pct(amp_f)}")
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

    parts.append("noisy" if bool(noisy) else "noiseless")

    parts.extend(
        _noise_tokens(
            float(cfg.get("depolarizing_prob", 0.0)),
            float(cfg.get("amplitude_damping_prob", 0.0)),
        )
    )

    parts.append(f"s{int(seed)}")
    parts.append(str(hash_str).strip())

    return "_".join(parts)
