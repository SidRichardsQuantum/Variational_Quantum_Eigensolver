import numpy as np
from vqe.core import run_vqe
from vqe.hamiltonian import build_h2_hamiltonian

def test_vqe_runs_minimal():
    # Tiny molecule (H2)
    H = build_h2_hamiltonian(bond_length=0.7)

    # Use simplified VQE settings
    result = run_vqe(
        hamiltonian=H,
        ansatz="TwoQubit-RY-CNOT",
        optimizer="adam",
        steps=3,
        seed=0,
        verbose=False
    )

    assert "final_energy" in result
    assert np.isfinite(result["final_energy"])
