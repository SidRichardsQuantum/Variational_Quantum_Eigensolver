"""
vqe_qpe_common.plotting
================

Centralised plotting utilities for the entire VQE/QPE package.

Provides:
    - Unified filename builder for all PNG plots
    - Automatic path sanitisation (molecule names, operators, symbols)
    - Consistent plotting export (DPI, bbox, tight layout)
    - Single point of control for changing any plotting behaviour

This ensures:
    • Zero filename collisions
    • Fully consistent naming between VQE, QPE, and notebooks
    • Clean, readable filenames for publication-quality figures
"""

from __future__ import annotations
import os
from typing import Dict, Optional
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(BASE_DIR, "images")


def ensure_plot_dirs(subdir: Optional[str] = None) -> str:
    """Ensure required directories exist and return the target directory."""
    base = IMG_DIR
    if subdir:
        base = os.path.join(IMG_DIR, format_molecule_name(subdir))
    os.makedirs(base, exist_ok=True)
    return base


# ---------------------------------------------------------------------
# Name Sanitisation
# ---------------------------------------------------------------------
def format_molecule_name(mol: str) -> str:
    """
    Convert a molecule name into a filesystem-safe token.

    Examples:
        "H3+"       → "H3plus"
        "H2 O"      → "H2_O"
        "LiH"       → "LiH"
    """
    mol = mol.replace("+", "plus")
    mol = mol.replace(" ", "_")
    return mol


def format_token(val: Optional[str | float | int]) -> Optional[str]:
    """
    Convert metadata values into clean filename components.

    Examples:
        0.05      → "0p05"
        0.0       → "0p0"
        "Adam"    → "Adam"
        None      → None
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        s = f"{val:.5f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    return str(val).replace(" ", "_")


# ---------------------------------------------------------------------
# Filename Builder
# ---------------------------------------------------------------------
def build_filename(
    molecule: Optional[str] = None,
    *,
    topic: str,
    extras: Optional[Dict[str, Optional[float | int | str]]] = None,
) -> str:
    """
    Build a descriptive, collision-safe PNG filename.

    Parameters
    ----------
    molecule : str, optional
        Molecule identifier (e.g., "H2", "H3+").
    topic : str
        The major theme of the plot, e.g.
            "vqe_convergence", "qpe_distribution",
            "noise_sweep", "optimizer_comparison"
    extras : dict, optional
        Arbitrary metadata to encode in filename, e.g.
            {"optimizer": "Adam", "ansatz": "UCCSD", "anc": 4}

    Returns
    -------
    str
        Filename ending in `.png`.

    Example
    -------
        build_filename(
            molecule="H2",
            topic="qpe_distribution",
            extras={"anc": 4, "pdep": 0.02}
        )

        → "H2_qpe_distribution_anc4_pdep0p02.png"
    """
    parts = []

    if molecule:
        parts.append(format_molecule_name(molecule))

    # Topic (safe)
    topic = topic.lower().replace(" ", "_")
    parts.append(topic)

    # Append metadata tokens
    if extras:
        for key, val in extras.items():
            fv = format_token(val)
            if fv is not None:
                parts.append(f"{key}{fv}")

    return "_".join(parts) + ".png"


# ---------------------------------------------------------------------
# Unified Save
# ---------------------------------------------------------------------
def save_plot(filename: str, show: bool = True, subdir: Optional[str] = None) -> str:
    target_dir = ensure_plot_dirs(subdir=subdir)

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
