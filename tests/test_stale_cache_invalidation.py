from __future__ import annotations

from pathlib import Path
from typing import Any

import pennylane as qml

from common.persist import atomic_write_json, read_json
from qite import run_qite, run_qrte
from qite.io_utils import RESULTS_DIR as QITE_RESULTS_DIR
from qpe import run_qpe
from qpe.io_utils import RESULTS_DIR as QPE_RESULTS_DIR
from vqe import run_vqe
from vqe.io_utils import RESULTS_DIR as VQE_RESULTS_DIR


def _single_qubit_model() -> qml.Hamiltonian:
    return qml.Hamiltonian([1.0], [qml.PauliZ(0)])


def _cache_file(results_dir: Path, label: str) -> Path:
    matches = sorted(results_dir.glob(f"*{label}*.json"))
    assert len(matches) == 1
    return matches[0]


def _remove_runtime_metadata(path: Path) -> None:
    payload = read_json(path)
    result: dict[str, Any]
    if isinstance(payload.get("result"), dict):
        result = payload["result"]
    else:
        result = payload

    result.pop("compute_runtime_s", None)
    result.pop("runtime_s", None)
    atomic_write_json(path, payload)


def test_vqe_recomputes_cache_record_without_runtime_metadata() -> None:
    label = "expert_vqe_stale_runtime_cache"
    cfg = dict(
        molecule=label,
        hamiltonian=_single_qubit_model(),
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        optimizer_name="Adam",
        steps=1,
        stepsize=0.1,
        plot=False,
        seed=201,
    )

    fresh = run_vqe(force=True, **cfg)
    path = _cache_file(VQE_RESULTS_DIR, label)
    _remove_runtime_metadata(path)

    refreshed = run_vqe(force=False, **cfg)
    record = read_json(path)

    assert fresh["cache_hit"] is False
    assert refreshed["cache_hit"] is False
    assert "compute_runtime_s" in record["result"]
    assert "runtime_s" in record["result"]


def test_qpe_recomputes_cache_record_without_runtime_metadata() -> None:
    label = "expert_qpe_stale_runtime_cache"
    cfg = dict(
        molecule=label,
        hamiltonian=_single_qubit_model(),
        hf_state=[1],
        n_ancilla=1,
        shots=None,
        plot=False,
        seed=202,
    )

    fresh = run_qpe(force=True, **cfg)
    path = _cache_file(QPE_RESULTS_DIR, label)
    _remove_runtime_metadata(path)

    refreshed = run_qpe(force=False, **cfg)
    record = read_json(path)

    assert fresh["cache_hit"] is False
    assert refreshed["cache_hit"] is False
    assert "compute_runtime_s" in record
    assert "runtime_s" in record


def test_qite_recomputes_cache_record_without_runtime_metadata() -> None:
    label = "expert_qite_stale_runtime_cache"
    cfg = dict(
        molecule=label,
        hamiltonian=_single_qubit_model(),
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        steps=1,
        dtau=0.1,
        plot=False,
        show=False,
        seed=203,
    )

    fresh = run_qite(force=True, **cfg)
    path = _cache_file(QITE_RESULTS_DIR, label)
    _remove_runtime_metadata(path)

    refreshed = run_qite(force=False, **cfg)
    record = read_json(path)

    assert fresh["cache_hit"] is False
    assert refreshed["cache_hit"] is False
    assert "compute_runtime_s" in record["result"]
    assert "runtime_s" in record["result"]


def test_qrte_recomputes_cache_record_without_runtime_metadata() -> None:
    label = "expert_qrte_stale_runtime_cache"
    cfg = dict(
        molecule=label,
        hamiltonian=_single_qubit_model(),
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        steps=1,
        dt=0.05,
        plot=False,
        show=False,
        seed=204,
    )

    fresh = run_qrte(force=True, **cfg)
    path = _cache_file(QITE_RESULTS_DIR, label)
    _remove_runtime_metadata(path)

    refreshed = run_qrte(force=False, **cfg)
    record = read_json(path)

    assert fresh["cache_hit"] is False
    assert refreshed["cache_hit"] is False
    assert "compute_runtime_s" in record["result"]
    assert "runtime_s" in record["result"]
