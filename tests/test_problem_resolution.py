from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from common.problem import ResolvedProblem, resolve_problem


def test_resolve_problem_expert_mode_normalizes_wires() -> None:
    H = qml.Hamiltonian([1.0], [qml.PauliZ("anc")])

    problem = resolve_problem(
        hamiltonian=H,
        num_qubits=1,
        reference_state=[1],
    )

    assert isinstance(problem, ResolvedProblem)
    assert problem.cacheable is False
    assert problem.num_qubits == 1
    assert list(problem.hamiltonian.wires) == [0]
    np.testing.assert_array_equal(problem.reference_state, np.array([1], dtype=int))


def test_resolve_problem_explicit_geometry_mode_returns_metadata() -> None:
    problem = resolve_problem(
        molecule="",
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]],
        charge=0,
        basis="sto-3g",
    )

    assert problem.cacheable is True
    assert problem.molecule_label == "molecule"
    assert problem.basis == "sto-3g"
    assert problem.charge == 0
    assert problem.num_qubits == 4
    assert len(problem.symbols) == 2
    assert problem.reference_state is not None


def test_resolve_problem_rejects_reference_without_hamiltonian() -> None:
    with pytest.raises(ValueError, match="reference_state is only supported"):
        resolve_problem(molecule="H2", reference_state=[1, 1])
