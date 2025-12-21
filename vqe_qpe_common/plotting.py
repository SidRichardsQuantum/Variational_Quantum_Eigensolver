"""
vqe_qpe_common.plotting
=======================

Centralised plotting utilities for the entire VQE/QPE package.

Guarantee:
- All PNG outputs are routed to:
    images/vqe/   for VQE plots
    images/qpe/   for QPE plots

Callers MUST specify kind="vqe" or kind="qpe" when saving.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_ROOT = os.path.join(BASE_DIR, "images")


def format_molecule_name(mol: str) -> str:
    mol = mol.replace("+", "plus")
    mol = mol.replace(" ", "_")
    return mol


def format_token(val: Optional[str | float | int]) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        s = f"{val:.5f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    return str(val).replace(" ", "_")


def build_filename(
    molecule: Optional[str] = None,
    *,
    topic: str,
    extras: Optional[Dict[str, Optional[float | int | str]]] = None,
) -> str:
    parts = []

    if molecule:
        parts.append(format_molecule_name(molecule))

    topic = topic.lower().replace(" ", "_")
    parts.append(topic)

    if extras:
        for key, val in extras.items():
            fv = format_token(val)
            if fv is not None:
                parts.append(f"{key}{fv}")

    return "_".join(parts) + ".png"


def _kind_dir(kind: str) -> str:
    k = str(kind).strip().lower()
    if k not in {"vqe", "qpe"}:
        raise ValueError(f"kind must be 'vqe' or 'qpe' (got {kind!r})")
    return os.path.join(IMG_ROOT, k)


def ensure_plot_dirs(*, kind: str) -> str:
    target = _kind_dir(kind)
    os.makedirs(target, exist_ok=True)
    return target


def save_plot(filename: str, *, kind: str, show: bool = True) -> str:
    target_dir = ensure_plot_dirs(kind=kind)

    if not filename.lower().endswith(".png"):
        filename = filename + ".png"

    path = os.path.join(target_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"📁 Saved plot → {path}")
    return path
