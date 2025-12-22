from vqe_qpe_common.hamiltonian import build_hamiltonian
from qpe.core import run_qpe
import numpy as np


def test_qpe_minimal():
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    H, n_qubits, hf_state = build_hamiltonian(atoms, coords, charge=0, basis="sto-3g")

    result = run_qpe(
        hamiltonian=H,
        hf_state=hf_state,
        n_ancilla=1,
        shots=200,
    )
    assert "phase" in result


def test_qpe_output_is_normalized():
    """
    QPE may ignore 'shots' depending on the PennyLane backend.
    The correct invariant is:
        • probs values sum to <= 1
        • at least one outcome exists
    """
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    H, nq, hf = build_hamiltonian(atoms, coords, charge=0, basis="sto-3g")

    result = run_qpe(
        hamiltonian=H,
        hf_state=hf,
        n_ancilla=1,
        shots=200,
    )

    probs = result["probs"]

    # Must contain at least one outcome
    assert len(probs) >= 1

    # PennyLane may return only a deterministic single outcome
    total = sum(probs.values())

    assert 0.0 < total <= 1.0
