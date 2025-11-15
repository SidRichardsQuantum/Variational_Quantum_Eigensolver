"""
qpe/io_utils.py
================
Handles result persistence, hashing, and image-saving utilities
for Quantum Phase Estimation (QPE).

This version fully synchronizes all plotting and filename conventions
with the shared `common.plotting` system used throughout the project.
"""

from __future__ import annotations
import os
import json
import hashlib
from typing import Any, Dict

from common.plotting import save_plot, build_filename


# ---------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "package_results")
IMG_DIR = os.path.join(BASE_DIR, "qpe", "images")


# ---------------------------------------------------------------------
# Directory Management
# ---------------------------------------------------------------------
def ensure_dirs() -> None:
    """Ensure results and image directories exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Hashing / Caching Utilities
# ---------------------------------------------------------------------
def signature_hash(
    molecule: str,
    n_ancilla: int,
    t: float,
    noise: bool,
    shots: int | None,
) -> str:
    """Generate a reproducible short hash for a given QPE configuration."""
    key = json.dumps(
        {
            "molecule": molecule,
            "ancilla_qubits": n_ancilla,
            "time_param": round(float(t), 8),
            "noise_enabled": bool(noise),
            "shots": shots,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def cache_path(molecule: str, hash_key: str) -> str:
    """Return the canonical JSON results file path."""
    ensure_dirs()
    safe_mol = molecule.replace("+", "plus").replace(" ", "_")
    return os.path.join(RESULTS_DIR, f"{safe_mol}_QPE_{hash_key}.json")


# ---------------------------------------------------------------------
# Result Persistence
# ---------------------------------------------------------------------
def save_qpe_result(result: Dict[str, Any]) -> str:
    """
    Save a QPE result to JSON in `package_results/`.

    Returns:
        Full path to the saved JSON result file.
    """
    ensure_dirs()
    key = signature_hash(
        molecule=result["molecule"],
        n_ancilla=result["n_ancilla"],
        t=result["t"],
        noise=bool(result.get("noise")),
        shots=result.get("shots", None),
    )
    path = cache_path(result["molecule"], key)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"💾 Saved QPE result → {path}")
    return path


def load_qpe_result(molecule: str, hash_key: str) -> Dict[str, Any] | None:
    """Load a cached QPE JSON result if it exists; otherwise return None."""
    path = cache_path(molecule, hash_key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------
# Unified Plot Saving (PNG)
# ---------------------------------------------------------------------
def save_qpe_plot(filename: str) -> str:
    """
    Save a QPE plot using the unified project-wide PNG logic.

    Parameters
    ----------
    filename : str
        A filename produced by `build_filename()` or a simple string.
        Example: "H2_QPE_distribution_4q.png"

    Returns
    -------
    str
        Full path where the PNG was saved.
    """
    ensure_dirs()
    return save_plot(filename)
