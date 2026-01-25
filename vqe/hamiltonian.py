"""
vqe.hamiltonian
---------------
VQE-facing Hamiltonian and geometry utilities.

This module is a thin compatibility layer over common. It preserves the
historical VQE API:

    - MOLECULES
    - generate_geometry(...)
    - build_hamiltonian(molecule, mapping="jordan_wigner")

returning:
    (H, qubits, hf_state, symbols, coordinates, basis, charge, unit)

Single source of truth:
    - molecule registry:    common.molecules
    - geometry generators:  common.geometry
    - Hamiltonian builder:  common.hamiltonian.build_hamiltonian
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml

from common.geometry import generate_geometry as _common_generate_geometry
from common.hamiltonian import build_hamiltonian as _build_common_hamiltonian
from common.molecules import MOLECULES as _COMMON_MOLECULES

# ---------------------------------------------------------------------
# Public re-export: molecule registry (backwards compatible)
# ---------------------------------------------------------------------
MOLECULES = _COMMON_MOLECULES


# ---------------------------------------------------------------------
# Compatibility: parametric geometry generation
# ---------------------------------------------------------------------
def generate_geometry(
    molecule: str, param_value: float
) -> Tuple[list[str], np.ndarray]:
    """
    Compatibility wrapper.

    Delegates to common.geometry.generate_geometry (single source of truth).
    """
    name = str(molecule).strip()
    return _common_generate_geometry(name, float(param_value))


def build_hamiltonian(
    molecule: str,
    mapping: Optional[str] = "jordan_wigner",
    unit: str = "angstrom",
) -> Tuple[qml.Hamiltonian, int, np.ndarray, List[str], np.ndarray, str, int, str]:
    (
        H,
        n_qubits,
        hf_state,
        symbols,
        coordinates,
        basis,
        charge,
        unit_out,
    ) = _build_common_hamiltonian(
        molecule=str(molecule),
        mapping=(
            str(mapping).strip().lower() if mapping is not None else "jordan_wigner"
        ),
        unit=str(unit).strip().lower(),
        return_metadata=True,
    )

    return (
        H,
        int(n_qubits),
        np.array(hf_state, dtype=int),
        list(symbols),
        np.array(coordinates, dtype=float),
        str(basis).strip().lower(),
        int(charge),
        str(unit_out),
    )


def build_hamiltonian_from_geometry(
    *,
    symbols: list[str],
    coordinates: np.ndarray,
    charge: int,
    basis: str,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
) -> Tuple[qml.Hamiltonian, int, np.ndarray, List[str], np.ndarray, str, int, str]:
    (
        H,
        n_qubits,
        hf_state,
        sym_out,
        coords_out,
        basis_out,
        charge_out,
        unit_out,
    ) = _build_common_hamiltonian(
        symbols=list(symbols),
        coordinates=np.array(coordinates, dtype=float),
        charge=int(charge),
        basis=str(basis),
        mapping=str(mapping).strip().lower(),
        unit=str(unit).strip().lower(),
        return_metadata=True,
    )

    return (
        H,
        int(n_qubits),
        np.array(hf_state, dtype=int),
        list(sym_out),
        np.array(coords_out, dtype=float),
        str(basis_out).strip().lower(),
        int(charge_out),
        str(unit_out),
    )


def hartree_fock_state(
    molecule: str,
    *,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
) -> np.ndarray:
    """
    Return the Hartree–Fock occupation bitstring for the molecule.
    """
    H, qubits, hf_state, symbols, coordinates, basis, charge, unit_out = (
        build_hamiltonian(
            molecule=molecule,
            mapping=mapping,
            unit=unit,
        )
    )
    return np.array(hf_state, dtype=int)
