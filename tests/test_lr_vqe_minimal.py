"""
tests.test_lr_vqe_minimal

Test the LR-VQE implementation on a minimal example (H2 molecule, minimal basis, small number of iterations) to check for basic sanity and determinism given a fixed random seed and force=True.
"""

from __future__ import annotations

import numpy as np


def test_lr_vqe_h2_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))

    from vqe.lr_vqe import run_lr_vqe

    cfg = dict(
        molecule="H2",
        k=2,
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=8,
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        fd_eps=1e-3,
        eps=1e-10,
        force=True,
    )

    r1 = run_lr_vqe(**cfg)
    r2 = run_lr_vqe(**cfg)

    exc1 = np.asarray(r1["excitations"], dtype=float)
    exc2 = np.asarray(r2["excitations"], dtype=float)

    assert exc1.shape == exc2.shape
    assert np.all(np.isfinite(exc1))
    assert np.allclose(exc1, exc2, atol=1e-8, rtol=0.0)
