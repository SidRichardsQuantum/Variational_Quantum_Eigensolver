from vqe_qpe_common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe
import numpy as np

def test_qpe_minimal():
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.7]])

    H, n_qubits, hf_state = build_hamiltonian(
        atoms, coords, charge=0, basis="sto-3g"
    )

    result = run_qpe(
        hamiltonian=H,
        hf_state=hf_state,
        n_ancilla=1,
        shots=100
    )

    assert "phase" in result
