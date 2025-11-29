"""
common.hamiltonian.py
==================

Shared molecular configuration + Hamiltonian construction for QPE.

This module mirrors the VQE architecture and ensures that QPE uses
the *exact same Hamiltonian pipeline* as VQE, guaranteeing consistent
chemistry, reproducibility, and shared geometry generation.

Provides:
    • get_molecule_config(name)
    • generate_geometry(name, param)
    • build_hamiltonian(symbols, coordinates, charge, basis)
"""

from __future__ import annotations
import numpy as np
import pennylane as qml
from pennylane import qchem
from typing import Dict, Any, Tuple

# ---------------------------------------------------------------------
# Import from VQE: SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------
from vqe.hamiltonian import (
    MOLECULES as VQE_MOLECULES,
    generate_geometry as vqe_generate_geometry,
)

# Expose VQE molecule registry to QPE
MOLECULES: Dict[str, Dict[str, Any]] = VQE_MOLECULES


def get_molecule_config(name: str) -> Dict[str, Any]:
    """
    Retrieve molecular configuration from the unified registry.
    """
    if name not in MOLECULES:
        raise KeyError(
            f"Unknown molecule '{name}'. Available: {list(MOLECULES.keys())}"
        )
    return MOLECULES[name]


# ---------------------------------------------------------------------
# Geometry variation (bond scans, angles)
# ---------------------------------------------------------------------
def generate_geometry(name: str, param: float):
    """
    Geometry generation wrapper for QPE.

    Calls the VQE geometry generator directly.
    """
    return vqe_generate_geometry(name, param)


# ---------------------------------------------------------------------
# Hamiltonian Builder
# ---------------------------------------------------------------------
def build_hamiltonian(
    symbols: list[str],
    coordinates: np.ndarray,
    charge: int,
    basis: str,
) -> Tuple[qml.Hamiltonian, int]:
    """
    Build the molecular Hamiltonian using PennyLane-qchem,
    with OpenFermion fallback.
    """
    try:
        H, n_qubits = qchem.molecular_hamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            charge=charge,
            basis=basis,
        )
        return H, n_qubits

    except Exception as e_primary:
        print("⚠️ PennyLane-qchem failed — retrying with OpenFermion backend...")

        try:
            H, n_qubits = qchem.molecular_hamiltonian(
                symbols=symbols,
                coordinates=coordinates,
                charge=charge,
                basis=basis,
                method="openfermion",
            )
            return H, n_qubits

        except Exception as e_fallback:
            raise RuntimeError(
                "Failed to construct Hamiltonian.\n"
                f"Primary error:\n  {e_primary}\n"
                f"OpenFermion fallback error:\n  {e_fallback}\n"
                "Try installing OpenFermion:\n"
                "    pip install openfermion openfermionpyscf\n"
            )
