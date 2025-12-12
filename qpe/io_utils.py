"""
qpe/io_utils.py
================
Unified result persistence, caching, and filename utilities for QPE.

This version is fully aligned with the VQE I/O stack:

    results/
      ├── vqe/
      └── qpe/
    
All QPE JSON output files live in: results/qpe/

All PNG figures must be saved through common.plotting.save_plot(),
which stores everything under: plots/

This file intentionally contains:
    • No plotting
    • No PennyLane logic
    • Only JSON I/O + persistent directory management
"""

from __future__ import annotations
import os
import json
import hashlib
from typing import Any, Dict

from vqe_qpe_common.plotting import save_plot, build_filename


# ---------------------------------------------------------------------
# Base Directories (mirrors VQE)
# ---------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "qpe")


def ensure_dirs() -> None:
    """Ensure that the QPE results directory exists."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Hashing: repeatable, human-stable, JSON-safe
# ---------------------------------------------------------------------
def signature_hash(
    *,
    molecule: str,
    n_ancilla: int,
    t: float,
    shots: int | None,
    noise: Dict[str, float] | None,
    trotter_steps: int,
) -> str:
    """
    Generate a reproducible hash key for a QPE configuration.

    All important run parameters are included.
    """
    key = json.dumps(
        {
            "molecule": molecule,
            "ancilla_qubits": n_ancilla,
            "time_param": round(float(t), 10),
            "trotter_steps": int(trotter_steps),
            "shots": shots,
            "noise_params": noise or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def cache_path(molecule: str, key: str) -> str:
    """Create the canonical JSON file path for a cached QPE result."""
    ensure_dirs()
    safe_mol = molecule.replace("+", "plus").replace(" ", "_")
    return os.path.join(RESULTS_DIR, f"{safe_mol}_QPE_{key}.json")


# ---------------------------------------------------------------------
# JSON Save / Load
# ---------------------------------------------------------------------
def save_qpe_result(result: Dict[str, Any]) -> str:
    """
    Save a QPE result to JSON using the canonical naming convention.

    Fields that MUST be present in result:
        molecule, n_ancilla, t, shots, noise
    """
    ensure_dirs()

    key = signature_hash(
        molecule=result["molecule"],
        n_ancilla=result["n_ancilla"],
        t=result["t"],
        trotter_steps=int(result.get("trotter_steps", 1)),
        shots=result.get("shots", None),
        noise=result.get("noise", {}),
    )

    path = cache_path(result["molecule"], key)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"💾 Saved QPE result → {path}")
    return path


def load_qpe_result(molecule: str, key: str) -> Dict[str, Any] | None:
    """
    Load a cached QPE JSON result if it exists, otherwise return None.
    """
    path = cache_path(molecule, key)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------
# Unified PNG Save Wrapper
# ---------------------------------------------------------------------
def save_qpe_plot(filename: str) -> str:
    """
    Save a QPE plot using the unified project-wide plotting logic.

    This stores all images under the root-level plots/ directory.

    filename should be produced by build_filename().
    """
    return save_plot(filename)
