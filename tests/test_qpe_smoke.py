from __future__ import annotations

import numpy as np

from common.hamiltonian import build_hamiltonian
from qpe import run_qpe


def test_qpe_minimal_smoke() -> None:
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    hamiltonian, _, hf_state = build_hamiltonian(
        atoms,
        coords,
        charge=0,
        basis="sto-3g",
    )

    res = run_qpe(
        hamiltonian=hamiltonian,
        hf_state=hf_state,
        n_ancilla=1,
        shots=200,
    )

    assert isinstance(res, dict)
    assert "phase" in res
    assert "probs" in res


def test_qpe_probability_dict_has_mass() -> None:
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    hamiltonian, _, hf_state = build_hamiltonian(
        atoms,
        coords,
        charge=0,
        basis="sto-3g",
    )

    res = run_qpe(
        hamiltonian=hamiltonian,
        hf_state=hf_state,
        n_ancilla=1,
        shots=200,
    )

    probs = res["probs"]
    assert isinstance(probs, dict)
    assert len(probs) >= 1

    total = sum(float(v) for v in probs.values())
    assert 0.0 < total <= 1.0
