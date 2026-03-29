from __future__ import annotations

import numpy as np

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
