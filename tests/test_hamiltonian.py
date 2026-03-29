from __future__ import annotations

from common.hamiltonian import build_hamiltonian
from common.molecules import get_molecule_config


def test_build_hamiltonian_h2() -> None:
    cfg = get_molecule_config("H2")
    hamiltonian, n_qubits, hf_state = build_hamiltonian(**cfg)

    assert n_qubits > 0
    assert len(hf_state) == n_qubits

    coeffs, ops = hamiltonian.terms()
    assert len(coeffs) == len(ops)
    assert len(coeffs) > 0
    assert all(hasattr(op, "wires") for op in ops)


def test_build_hamiltonian_selected_registry_molecules() -> None:
    for name in ["H2", "H3+", "LiH", "BeH2", "HeH+"]:
        cfg = get_molecule_config(name)
        hamiltonian, n_qubits, hf_state = build_hamiltonian(**cfg)

        assert n_qubits > 0
        assert len(hf_state) == n_qubits
        assert len(hamiltonian) > 0
