from __future__ import annotations

import numpy as np
import pennylane as qml

from common.persist import canonical_hamiltonian
from qite import run_qite, run_qrte
from qpe import run_qpe
from vqe import run_vqe
from vqe.auto_ansatz import resolve_auto_ansatz


def _single_qubit_model() -> qml.Hamiltonian:
    return qml.Hamiltonian([1.0], [qml.PauliZ(0)])


def _xxz_model() -> qml.Hamiltonian:
    return qml.Hamiltonian(
        [1.0, 1.0, 0.5],
        [
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliY(0) @ qml.PauliY(1),
            qml.PauliZ(0) @ qml.PauliZ(1),
        ],
    )


def _ssh_model() -> qml.Hamiltonian:
    return qml.Hamiltonian(
        [0.5, 0.5],
        [
            qml.PauliX(0) @ qml.PauliX(1),
            qml.PauliY(0) @ qml.PauliY(1),
        ],
    )


def _fallback_model() -> qml.Hamiltonian:
    return qml.Hamiltonian([0.75], [qml.PauliX(0) @ qml.PauliY(1)])


def test_canonical_hamiltonian_is_stable_for_equivalent_pauli_terms() -> None:
    hamiltonian = qml.Hamiltonian(
        [1.0, 0.25, 0.0],
        [
            qml.PauliZ(5),
            qml.PauliX(3) @ qml.PauliX(5),
            qml.PauliY(3),
        ],
    )
    reordered = qml.Hamiltonian(
        [0.25, 1.0],
        [
            qml.PauliX(5) @ qml.PauliX(3),
            qml.PauliZ(5),
        ],
    )

    assert canonical_hamiltonian(hamiltonian) == canonical_hamiltonian(reordered)


def test_auto_ansatz_detects_expert_model_families() -> None:
    xxz_name, xxz_kwargs, xxz_meta = resolve_auto_ansatz(
        "auto", _xxz_model(), 2, ansatz_kwargs={"layers": 2}
    )
    ssh_name, ssh_kwargs, ssh_meta = resolve_auto_ansatz("auto", _ssh_model(), 2)
    fallback_name, fallback_kwargs, fallback_meta = resolve_auto_ansatz(
        "auto", _fallback_model(), 2
    )

    assert xxz_name == "XXZ-HVA"
    assert xxz_kwargs == {"layers": 2}
    assert xxz_meta is not None
    assert xxz_meta["selected"] == "XXZ-HVA"

    assert ssh_name == "NumberPreservingGivens"
    assert ssh_kwargs == {"layers": 3}
    assert ssh_meta is not None
    assert ssh_meta["selected"] == "NumberPreservingGivens"

    assert fallback_name == "StronglyEntanglingLayers"
    assert fallback_kwargs == {"layers": 2}
    assert fallback_meta is not None
    assert fallback_meta["selected"] == "StronglyEntanglingLayers"


def test_vqe_expert_auto_ansatz_fallback_reports_selection_metadata() -> None:
    res = run_vqe(
        molecule="expert_auto_fallback_metadata",
        hamiltonian=_fallback_model(),
        num_qubits=2,
        reference_state=[0, 0],
        ansatz_name="auto",
        ansatz_kwargs={"layers": 1},
        optimizer_name="Adam",
        steps=1,
        stepsize=0.1,
        plot=False,
        force=True,
        seed=0,
    )

    assert np.isfinite(float(res["energy"]))
    assert res["ansatz"] == "StronglyEntanglingLayers"
    assert res["ansatz_kwargs"] == {"layers": 1}
    assert res["ansatz_selection"]["requested"] == "auto"
    assert res["ansatz_selection"]["selected"] == "StronglyEntanglingLayers"


def test_vqe_expert_cache_key_uses_canonical_hamiltonian_fingerprint() -> None:
    cfg = dict(
        molecule="expert_vqe_fingerprint_cache",
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        optimizer_name="Adam",
        steps=1,
        stepsize=0.1,
        plot=False,
        seed=101,
    )

    fresh = run_vqe(force=True, hamiltonian=_single_qubit_model(), **cfg)
    cached = run_vqe(
        force=False,
        hamiltonian=qml.Hamiltonian([1.0], [qml.PauliZ(3)]),
        **cfg,
    )

    assert fresh["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert np.isclose(float(cached["compute_runtime_s"]), fresh["compute_runtime_s"])


def test_qpe_expert_cache_key_uses_canonical_hamiltonian_fingerprint() -> None:
    cfg = dict(
        molecule="expert_qpe_fingerprint_cache",
        hf_state=[1],
        n_ancilla=1,
        shots=None,
        plot=False,
        seed=102,
    )

    fresh = run_qpe(force=True, hamiltonian=_single_qubit_model(), **cfg)
    cached = run_qpe(
        force=False,
        hamiltonian=qml.Hamiltonian([1.0], [qml.PauliZ(3)]),
        **cfg,
    )

    assert fresh["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert np.isclose(float(cached["compute_runtime_s"]), fresh["compute_runtime_s"])


def test_qite_expert_cache_key_uses_canonical_hamiltonian_fingerprint() -> None:
    cfg = dict(
        molecule="expert_qite_fingerprint_cache",
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        steps=1,
        dtau=0.1,
        plot=False,
        show=False,
        seed=103,
    )

    fresh = run_qite(force=True, hamiltonian=_single_qubit_model(), **cfg)
    cached = run_qite(
        force=False,
        hamiltonian=qml.Hamiltonian([1.0], [qml.PauliZ(3)]),
        **cfg,
    )

    assert fresh["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert np.isclose(float(cached["compute_runtime_s"]), fresh["compute_runtime_s"])


def test_qrte_expert_cache_key_uses_canonical_hamiltonian_fingerprint() -> None:
    cfg = dict(
        molecule="expert_qrte_fingerprint_cache",
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        steps=1,
        dt=0.05,
        plot=False,
        show=False,
        seed=104,
    )

    fresh = run_qrte(force=True, hamiltonian=_single_qubit_model(), **cfg)
    cached = run_qrte(
        force=False,
        hamiltonian=qml.Hamiltonian([1.0], [qml.PauliZ(3)]),
        **cfg,
    )

    assert fresh["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert np.isclose(float(cached["compute_runtime_s"]), fresh["compute_runtime_s"])
