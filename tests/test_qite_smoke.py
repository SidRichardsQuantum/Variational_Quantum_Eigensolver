from __future__ import annotations

import numpy as np
import pennylane as qml
import pytest

from qite import run_qite, run_qrte
from qite.engine import build_ansatz
from vqe import run_vqe


def test_qite_minimal_smoke() -> None:
    res = run_qite(
        molecule="H2",
        ansatz_name="UCCSD",
        steps=4,
        dtau=0.2,
        force=True,
        plot=False,
        show=False,
    )

    assert isinstance(res, dict)
    assert "energy" in res
    assert "energies" in res
    assert "num_qubits" in res
    assert "varqite" in res
    assert "runtime_s" in res
    assert "compute_runtime_s" in res
    assert "cache_hit" in res
    assert "environment" in res

    assert np.isfinite(float(res["energy"]))
    assert len(res["energies"]) >= 1
    assert int(res["num_qubits"]) > 0
    assert res["environment"]["python"]
    assert "pennylane" in res["environment"]["packages"]


def test_qite_uccsd_descends_for_h2() -> None:
    res = run_qite(
        molecule="H2",
        ansatz_name="UCCSD",
        steps=6,
        dtau=0.2,
        force=True,
        plot=False,
        show=False,
    )

    energies = np.asarray(res["energies"], dtype=float)
    assert energies.shape[0] >= 2
    assert float(energies[-1]) < float(energies[0]) - 1e-3


def test_qite_rejects_noisy_optimization() -> None:
    with pytest.raises(ValueError):
        run_qite(
            molecule="H2",
            steps=2,
            dtau=0.2,
            noisy=True,
            force=True,
            plot=False,
            show=False,
        )


def test_qite_rejects_unknown_ansatz_name() -> None:
    with pytest.raises(ValueError, match="Unknown ansatz"):
        run_qite(
            molecule="H2",
            ansatz_name="Adam",
            steps=1,
            dtau=0.2,
            force=True,
            plot=False,
            show=False,
        )


def test_qite_charged_uccsd_smoke() -> None:
    res = run_qite(
        molecule="H3+",
        ansatz_name="UCCSD",
        steps=1,
        dtau=0.1,
        force=True,
        plot=False,
        show=False,
    )

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert len(res["final_params"]) >= 1


def test_qite_prebuilt_hamiltonian_smoke_and_cache_hit() -> None:
    H = qml.Hamiltonian([1.0], [qml.PauliZ(4)])

    cfg = dict(
        molecule="expert_qite_cache_smoke",
        hamiltonian=H,
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        steps=2,
        dtau=0.1,
        plot=False,
        show=False,
    )

    fresh = run_qite(force=True, **cfg)
    res = run_qite(force=False, **cfg)

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert int(res["num_qubits"]) == 1
    assert fresh["cache_hit"] is False
    assert res["cache_hit"] is True
    assert "environment" in fresh
    assert "environment" in res


def test_qite_routes_ansatz_kwargs_to_non_molecule_ansatz() -> None:
    H = qml.Hamiltonian([1.0], [qml.PauliZ(0)])

    res = run_qite(
        hamiltonian=H,
        num_qubits=2,
        reference_state=[1, 0],
        ansatz_name="NumberPreservingGivens",
        ansatz_kwargs={"layers": 2},
        steps=1,
        dtau=0.1,
        force=True,
        plot=False,
        show=False,
        seed=0,
    )

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert int(res["num_qubits"]) == 2
    assert res["final_params_shape"] == [2, 1]


def test_qite_routes_model_specific_ansatz_kwargs() -> None:
    H = qml.Hamiltonian(
        [1.0, 1.0, 0.5],
        [
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliY(0) @ qml.PauliY(1),
            qml.PauliZ(0) @ qml.PauliZ(1),
        ],
    )

    res = run_qite(
        hamiltonian=H,
        num_qubits=2,
        reference_state=[1, 0],
        ansatz_name="XXZ-HVA",
        ansatz_kwargs={"layers": 2},
        steps=1,
        dtau=0.1,
        force=True,
        plot=False,
        show=False,
        seed=0,
    )

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert res["final_params_shape"] == [2, 4]


def test_qite_auto_ansatz_selects_xxz_hva() -> None:
    H = qml.Hamiltonian(
        [1.0, 1.0, 0.5],
        [
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliY(0) @ qml.PauliY(1),
            qml.PauliZ(0) @ qml.PauliZ(1),
        ],
    )

    res = run_qite(
        hamiltonian=H,
        num_qubits=2,
        reference_state=[1, 0],
        ansatz_name="auto",
        ansatz_kwargs={"layers": 2},
        steps=1,
        dtau=0.1,
        force=True,
        plot=False,
        show=False,
        seed=0,
    )

    assert res["ansatz"] == "XXZ-HVA"
    assert res["ansatz_kwargs"] == {"layers": 2}
    assert res["ansatz_selection"]["selected"] == "XXZ-HVA"
    assert res["final_params_shape"] == [2, 4]


def test_qite_auto_ansatz_selects_number_preserving_givens() -> None:
    H = qml.Hamiltonian(
        [0.5, 0.5],
        [qml.PauliX(0) @ qml.PauliX(1), qml.PauliY(0) @ qml.PauliY(1)],
    )

    res = run_qite(
        hamiltonian=H,
        num_qubits=2,
        reference_state=[1, 0],
        ansatz_name="auto",
        steps=1,
        dtau=0.1,
        force=True,
        plot=False,
        show=False,
        seed=0,
    )

    assert res["ansatz"] == "NumberPreservingGivens"
    assert res["ansatz_kwargs"] == {"layers": 3}
    assert res["ansatz_selection"]["selected"] == "NumberPreservingGivens"
    assert res["final_params_shape"] == [3, 1]


def test_qite_engine_delegates_charge_aware_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_vqe_build_ansatz(
        ansatz_name,
        num_wires,
        *,
        seed=0,
        symbols=None,
        coordinates=None,
        charge=0,
        basis="sto-3g",
        requires_grad=True,
        scale=0.01,
    ):
        captured["charge"] = charge
        captured["symbols"] = symbols
        captured["coordinates"] = np.array(coordinates, dtype=float)
        captured["basis"] = basis

        def ansatz_fn(params, wires, charge=0, **kwargs):
            captured["ansatz_charge"] = charge

        return ansatz_fn, []

    monkeypatch.setattr("vqe.engine.build_ansatz", fake_vqe_build_ansatz)

    ansatz_fn, params = build_ansatz(
        "UCCSD",
        1,
        seed=0,
        symbols=["H"],
        coordinates=[[0.0, 0.0, 0.0]],
        charge=1,
        basis="6-31g",
        requires_grad=False,
        hf_state=np.array([1], dtype=int),
    )

    ansatz_fn(params)

    assert captured["charge"] == 1
    assert captured["ansatz_charge"] == 1
    assert captured["symbols"] == ["H"]
    assert np.array_equal(
        captured["coordinates"], np.array([[0.0, 0.0, 0.0]], dtype=float)
    )
    assert captured["basis"] == "6-31g"


def test_qrte_minimal_smoke() -> None:
    res = run_qrte(
        molecule="H2",
        ansatz_name="UCCSD",
        steps=4,
        dt=0.05,
        force=True,
        plot=False,
        show=False,
    )

    assert isinstance(res, dict)
    assert "energy" in res
    assert "energies" in res
    assert "times" in res
    assert "params_history" in res
    assert "num_qubits" in res
    assert "varqrte" in res
    assert "runtime_s" in res
    assert "compute_runtime_s" in res
    assert "cache_hit" in res

    assert np.isfinite(float(res["energy"]))
    assert len(res["energies"]) >= 1
    assert len(res["times"]) == len(res["energies"])
    assert len(res["params_history"]) == len(res["energies"])
    assert abs(float(res["times"][-1]) - 0.2) < 1e-12
    assert int(res["num_qubits"]) > 0


def test_qite_cache_hit_reports_cached_timing_metadata() -> None:
    cfg = dict(
        molecule="H2",
        ansatz_name="UCCSD",
        steps=1,
        dtau=0.2,
        plot=False,
        show=False,
        seed=321,
    )

    fresh = run_qite(force=True, **cfg)
    cached = run_qite(force=False, **cfg)

    assert fresh["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert np.isclose(
        float(cached["compute_runtime_s"]),
        float(fresh["compute_runtime_s"]),
    )


def test_qrte_rejects_noisy_optimization() -> None:
    with pytest.raises(ValueError):
        run_qrte(
            molecule="H2",
            steps=2,
            dt=0.05,
            noisy=True,
            force=True,
            plot=False,
            show=False,
        )


def test_qrte_accepts_prepared_initial_params() -> None:
    prepared = run_vqe(
        molecule="H2",
        ansatz_name="StronglyEntanglingLayers",
        optimizer_name="Adam",
        steps=3,
        stepsize=0.1,
        plot=False,
        force=True,
    )

    res = run_qrte(
        molecule="H2",
        ansatz_name="StronglyEntanglingLayers",
        steps=1,
        dt=0.05,
        initial_params=prepared["final_params"],
        force=True,
        plot=False,
        show=False,
    )

    assert res["initialization"] == "provided"
    np.testing.assert_allclose(
        np.asarray(res["params_history"][0], dtype=float),
        np.asarray(prepared["final_params"], dtype=float),
    )
