import numpy as np
from qpe.core import run_qpe
from qpe.hamiltonian import build_h2_hamiltonian

def test_qpe_runs_minimal():
    H = build_h2_hamiltonian(0.7)

    result = run_qpe(
        hamiltonian=H,
        num_ancillas=2,
        time=0.2,
        shots=100,
        seed=0,
        verbose=False,
    )

    assert "phase_distribution" in result
    assert isinstance(result["phase_distribution"], dict)
