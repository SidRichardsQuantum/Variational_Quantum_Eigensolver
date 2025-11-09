"""
QPE I/O Utilities
=================
Handles file paths, caching, and result persistence for Quantum Phase Estimation (QPE).
Ensures consistent storage layout alongside the VQE package.
"""

from __future__ import annotations
import os
import json
import hashlib
from typing import Any, Dict

# ---------------------------------------------------------------------
# Base Directories (mirroring VQE)
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
    """Generate a short reproducible hash for a QPE configuration."""
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
    """Return the canonical cache file path for a given molecule and hash key."""
    ensure_dirs()
    safe_mol = molecule.replace("+", "plus").replace(" ", "_")
    return os.path.join(RESULTS_DIR, f"{safe_mol}_QPE_{hash_key}.json")


# ---------------------------------------------------------------------
# Result Persistence
# ---------------------------------------------------------------------
def save_qpe_result(result: Dict[str, Any]) -> str:
    """Save a QPE result to JSON in `package_results/`.

    Returns:
        The file path of the saved result.
    """
    ensure_dirs()
    key = signature_hash(
        result["molecule"],
        result["n_ancilla"],
        result["t"],
        result.get("noise", False),
        result.get("shots", None),
    )
    path = cache_path(result["molecule"], key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"💾 Saved QPE result → {path}")
    return path


def load_qpe_result(molecule: str, hash_key: str) -> Dict[str, Any] | None:
    """Load a cached QPE result if available; return None if not found."""
    path = cache_path(molecule, hash_key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------
# Plot Utilities
# ---------------------------------------------------------------------
def save_qpe_plot(name: str) -> str:
    """Return a full image save path in `qpe/images/` and ensure directory exists."""
    ensure_dirs()
    return os.path.join(IMG_DIR, name)
