"""
common.hamiltonian
==================

Shared Hamiltonian construction used by VQE, QPE, and future solvers.
"""

from __future__ import annotations
import numpy as np
import pennylane as qml
from pennylane import qchem
from typing import Tuple

from vqe_qpe_common.molecules import get_molecule_config

def hartree_fock_state(symbols, charge, n_qubits):
    """Compute electron count and return HF basis state bitstring."""
    Z = {"H": 1, "Li": 3, "O": 8}
    electrons = sum(Z[s] for s in symbols) - charge
    return qchem.hf_state(int(electrons), n_qubits)

def build_hamiltonian(symbols, coordinates, charge, basis) -> Tuple[qml.Hamiltonian, int, np.ndarray]:
    """
    Build Hamiltonian + deduce HF state.
    Returns: (Hamiltonian, n_qubits, hf_state)
    """
    try:
        H, n_qubits = qchem.molecular_hamiltonian(
            symbols=symbols,
            coordinates=coordinates,
            charge=charge,
            basis=basis,
        )
    except Exception as e_primary:
        print("⚠️ Default PennyLane-qchem backend failed — retrying with OpenFermion...")
        try:
            H, n_qubits = qchem.molecular_hamiltonian(
                symbols=symbols,
                coordinates=coordinates,
                charge=charge,
                basis=basis,
                method="openfermion",
            )
        except Exception as e_fallback:
            raise RuntimeError(
                f"Failed to construct Hamiltonian.\n"
                f"Primary error: {e_primary}\n"
                f"Fallback error: {e_fallback}"
            )

    hf_state = hartree_fock_state(symbols, charge, n_qubits)
    return H, n_qubits, hf_state
