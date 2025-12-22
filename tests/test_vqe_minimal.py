from vqe_qpe_common.hamiltonian import build_hamiltonian
from vqe.core import run_vqe
import numpy as np


def test_vqe_minimal():
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    H, n_qubits, hf_state = build_hamiltonian(atoms, coords, charge=0, basis="sto-3g")

    result = run_vqe("H2", ansatz_name="Minimal", optimizer_name="Adam", steps=2)

    assert "energy" in result
