"""
qite.hamiltonian
----------------
QITE-facing Hamiltonian utilities.

Thin compatibility layer over the shared Hamiltonian pipeline (common.hamiltonian).
Mirrors qpe.hamiltonian so QITE does not depend on vqe.* internals.

Returns
-------
(H, n_qubits, hf_state, symbols, coordinates, basis, charge, mapping_out, unit_out)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pennylane as qml

from common.hamiltonian import build_hamiltonian as _build_hamiltonian


def build_hamiltonian(
    molecule: str | None = None,
    coordinates: np.ndarray | None = None,
    symbols: list[str] | None = None,
    *,
    charge: int | None = None,
    basis: str | None = None,
    multiplicity: int | None = None,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
) -> Tuple[qml.Hamiltonian, int, np.ndarray, List[str], np.ndarray, str, int, str, str]:
    mapping_out = str(mapping).strip().lower()
    unit_in = str(unit).strip().lower()

    (
        H,
        n_qubits,
        hf_state,
        symbols_out,
        coordinates_out,
        basis_out,
        charge_out,
        unit_out,
    ) = _build_hamiltonian(
        molecule=molecule,
        symbols=symbols,
        coordinates=coordinates,
        charge=charge,
        multiplicity=multiplicity,
        basis=basis,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        mapping=mapping_out,
        unit=unit_in,
        return_metadata=True,
    )

    return (
        H,
        int(n_qubits),
        np.array(hf_state, dtype=int),
        list(symbols_out),
        np.array(coordinates_out, dtype=float),
        str(basis_out).strip().lower(),
        int(charge_out),
        mapping_out,
        str(unit_out).strip().lower(),
    )
