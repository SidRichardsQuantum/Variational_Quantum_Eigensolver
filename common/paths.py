"""
common.paths
"""

from __future__ import annotations

import os
from pathlib import Path

from common.naming import format_molecule_name


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    override = os.environ.get("VQE_PENNYLANE_DATA_DIR", "").strip()
    if override:
        p = Path(override).expanduser().resolve()
        if p.exists() and (not p.is_dir()):
            raise ValueError(
                f"VQE_PENNYLANE_DATA_DIR must point to a directory (got {str(p)!r})"
            )
        p.mkdir(parents=True, exist_ok=True)
        return p
    return project_root()


def results_root() -> Path:
    return data_root() / "results"


def images_root() -> Path:
    return data_root() / "images"


def results_dir(kind: str) -> Path:
    k = "".join(ch for ch in str(kind).strip().lower() if ch.isalnum() or ch == "_")
    if k not in {"vqe", "qpe", "qite"}:
        raise ValueError(f"kind must be 'vqe', 'qpe', or 'qite' (got {kind!r})")
    return results_root() / k


def images_dir(kind: str, molecule: str | None = None) -> Path:
    k = "".join(ch for ch in str(kind).strip().lower() if ch.isalnum() or ch == "_")
    if k not in {"vqe", "qpe", "qite"}:
        raise ValueError(f"kind must be 'vqe', 'qpe', or 'qite' (got {kind!r})")
    d = images_root() / k
    if molecule:
        d = d / format_molecule_name(molecule)
    return d
