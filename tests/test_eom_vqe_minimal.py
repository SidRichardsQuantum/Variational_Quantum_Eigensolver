"""
tests.test_eom_vqe_minimal

Minimal deterministic smoke test for EOM-VQE (tangent-space full response)
on H2. Uses a more expressive ansatz and enough VQE iterations to obtain a
stable reference so that positive excitation energies exist.
"""

from __future__ import annotations

import numpy as np


def test_eom_vqe_h2_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))

    from vqe.eom_vqe import run_eom_vqe

    cfg = dict(
        molecule="H2",
        k=2,
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        steps=25,
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        fd_eps=1e-3,
        eps=1e-10,
        omega_eps=1e-12,
        force=True,
        plot=False,
        show=False,
        save=False,
    )

    r1 = run_eom_vqe(**cfg)
    r2 = run_eom_vqe(**cfg)

    exc1 = np.asarray(r1["excitations"], dtype=float)
    exc2 = np.asarray(r2["excitations"], dtype=float)

    assert exc1.shape == exc2.shape
    assert exc1.size >= 1
    assert np.all(np.isfinite(exc1))
    assert np.all(exc1 > 0.0)

    # Determinism
    assert np.allclose(exc1, exc2, atol=1e-8, rtol=0.0)
