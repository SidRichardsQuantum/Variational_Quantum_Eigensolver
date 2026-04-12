from __future__ import annotations

import numpy as np
import pennylane as qml

import vqe.core as vqe_core
from vqe import run_vqe


def test_vqe_minimal_smoke() -> None:
    res = run_vqe(
        molecule="H2",
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=4,
        stepsize=0.2,
        noisy=False,
        force=True,
        plot=False,
    )

    assert isinstance(res, dict)
    assert "energy" in res
    assert "energies" in res
    assert "num_qubits" in res

    assert np.isfinite(float(res["energy"]))
    assert len(res["energies"]) >= 1
    assert int(res["num_qubits"]) > 0


def test_vqe_deterministic_given_seed_and_force() -> None:
    cfg = dict(
        molecule="H2",
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=4,
        stepsize=0.2,
        seed=0,
        noisy=False,
        force=True,
        plot=False,
    )

    r1 = run_vqe(**cfg)
    r2 = run_vqe(**cfg)

    e1 = np.asarray(r1["energies"], dtype=float)
    e2 = np.asarray(r2["energies"], dtype=float)

    assert e1.shape == e2.shape
    assert np.all(np.isfinite(e1))
    assert np.allclose(e1, e2, atol=1e-10, rtol=0.0)


def test_vqe_prebuilt_hamiltonian_smoke_and_bypass_cache(monkeypatch) -> None:
    H = qml.Hamiltonian([1.0], [qml.PauliZ(3)])

    def fail_load(_prefix):
        raise AssertionError("expert-mode VQE should not read from cache")

    def fail_save(_prefix, _record):
        raise AssertionError("expert-mode VQE should not save to cache")

    monkeypatch.setattr(vqe_core, "load_run_record", fail_load)
    monkeypatch.setattr(vqe_core, "save_run_record", fail_save)

    res = run_vqe(
        hamiltonian=H,
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        optimizer_name="Adam",
        steps=2,
        stepsize=0.1,
        plot=False,
        force=False,
    )

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert int(res["num_qubits"]) == 1
