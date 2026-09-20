"""Scientific outcomes, shared inputs, and state-preserving refinement."""

import importlib
import json

import numpy as np
import pennylane as qml
import pytest

from common.termination import optimize
from qite.core import run_qite
from vqe.core import run_vqe


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))


@pytest.mark.parametrize("runner", [run_vqe, run_qite])
def test_termination_initialization_and_cache(runner):
    problem = dict(
        hamiltonian=qml.Hamiltonian([1.0], [qml.Z(0)]),
        num_qubits=1,
        reference_state=[0],
        ansatz_name="RY-CZ",
        plot=False,
    )
    initial = runner(**problem, steps=0)
    shape = initial["final_params_shape"]
    params = np.zeros(shape)
    result = runner(
        **problem, steps=8, initial_params=params, energy_tol=1e-10, patience=2
    )
    assert result["termination"]["reason"] == "tolerance_satisfied"
    assert result["steps"] == 2
    assert len(result["energies"]) == len(result["params_history"]) == 3
    assert result["initialization"]["source"] == "supplied"
    assert result["energies"][0] == pytest.approx(1.0)
    cached = runner(
        **problem, steps=8, initial_params=params, energy_tol=1e-10, patience=2
    )
    assert cached["cache_hit"]
    assert cached["termination"] == result["termination"]
    budget = runner(**problem, steps=3, initial_params=params)
    assert budget["termination"]["reason"] == "budget_exhausted"
    assert budget["steps"] == 3
    with pytest.raises(ValueError, match="shape"):
        runner(**problem, initial_params=[0.0] * 17)
    with pytest.raises(ValueError, match="finite"):
        runner(**problem, initial_params=[np.nan])


@pytest.mark.parametrize(
    "runner,module,update_name",
    [
        (run_vqe, "vqe.core", "engine_build_optimizer"),
        (run_qite, "qite.core", "qite_step"),
    ],
)
def test_numerical_failure_is_serializable_and_cached(
    runner, module, update_name, monkeypatch
):
    class BadOptimizer:
        def step_and_cost(self, fn, p):
            return p * np.nan, 0

    replacement = (
        (lambda *a, **k: BadOptimizer())
        if runner is run_vqe
        else (lambda **k: k["params"] * np.nan)
    )
    monkeypatch.setattr(importlib.import_module(module), update_name, replacement)
    args = dict(
        hamiltonian=qml.Hamiltonian([1.0], [qml.Z(0)]),
        num_qubits=1,
        ansatz_name="RY-CZ",
        steps=2,
        plot=False,
    )
    result = runner(**args)
    assert result["termination"]["reason"] == "numerical_failure"
    assert result["termination"]["attempted_updates"] == 1
    assert result["steps"] == 0
    assert len(result["energies"]) == 1
    json.dumps(result, allow_nan=False)
    assert runner(**args)["termination"] == result["termination"]


def test_initial_nonfinite_and_observer_exceptions():
    _, energies, history, status = optimize(
        np.array([0.0]), lambda p: np.nan, lambda p: p, steps=2
    )
    assert status["reason"] == "numerical_failure"
    assert energies == history == []

    def observer(event):
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        optimize(
            np.array([0.0]),
            lambda p: 1.0,
            lambda p: p,
            steps=2,
            progress_callback=observer,
        )


@pytest.mark.parametrize(
    "name", ["vqd", "ssvqe", "qse", "eom_qse", "lr_vqe", "eom_vqe"]
)
def test_excited_expert_problem_and_cache_identity(name):
    runner = getattr(importlib.import_module("vqe." + name), "run_" + name)
    args = dict(
        molecule="model",
        hamiltonian=qml.Hamiltonian([1.0, 0.3], [qml.Z(1), qml.X(1)]),
        num_qubits=2,
        ansatz_name="RY-CZ",
        reference_state=[0, 1],
        steps=40,
    )
    if name in {"vqd", "ssvqe", "lr_vqe", "eom_vqe"}:
        args["plot"] = False
    result = runner(**args)
    assert result["config"]["num_qubits"] == 2
    assert result["config"]["reference_state"] == [0, 1]
    other = runner(**{**args, "reference_state": [0, 0]})
    assert other["config"] != result["config"]


def test_adapt_active_space_and_termination():
    from vqe.adapt import run_adapt_vqe

    args = dict(
        symbols=["H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.74]],
        active_electrons=2,
        active_orbitals=2,
        plot=False,
    )
    budget = run_adapt_vqe(**args, max_ops=0)
    assert budget["termination"]["reason"] == "operator_budget_exhausted"
    tolerance = run_adapt_vqe(**args, max_ops=2, grad_tol=1e3)
    assert tolerance["termination"]["reason"] == "tolerance_satisfied"
    assert tolerance["config"]["active_electrons"] == 2
    with pytest.raises(ValueError, match="chemistry"):
        run_adapt_vqe(hamiltonian=qml.Hamiltonian([1.0], [qml.Z(0)]))
