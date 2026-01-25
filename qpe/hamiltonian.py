"""
qpe.hamiltonian
---------------
QPE-facing Hamiltonian utilities.

Thin compatibility layer over common.hamiltonian.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pennylane as qml

from common.hamiltonian import build_hamiltonian as _common_build_hamiltonian


def build_hamiltonian(
    molecule: str,
    *,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
) -> Tuple[qml.Hamiltonian, int, np.ndarray, List[str], np.ndarray, str, int, str]:
    """
    Returns
    -------
    (H, n_qubits, hf_state, symbols, coordinates, basis, charge, unit_out)
    """
    mapping_out = str(mapping).strip().lower()
    unit_in = str(unit).strip().lower()

    (
        H,
        n_qubits,
        hf_state,
        symbols,
        coordinates,
        basis,
        charge,
        unit_out,
    ) = _common_build_hamiltonian(
        molecule=str(molecule),
        mapping=mapping_out,
        unit=unit_in,
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
