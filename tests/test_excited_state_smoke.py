from __future__ import annotations

import math

import numpy as np

from common.hamiltonian import get_exact_spectrum
from vqe import run_qse
from vqe.adapt import run_adapt_vqe
from vqe.eom_qse import run_eom_qse
from vqe.eom_vqe import run_eom_vqe
from vqe.lr_vqe import run_lr_vqe


def _is_sorted_non_decreasing(xs: list[float], tol: float = 1e-12) -> bool:
    return all(xs[i] <= xs[i + 1] + tol for i in range(len(xs) - 1))


def _nearest_diffs(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    out = []
    for v in values:
        out.append(float(np.min(np.abs(candidates - v))))
    return np.asarray(out, dtype=float)


def test_adapt_vqe_smoke() -> None:
    res = run_adapt_vqe(
        molecule="H2",
        pool="uccs",
        max_ops=4,
        grad_tol=1e-3,
        inner_steps=8,
        inner_stepsize=0.2,
        optimizer_name="Adam",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        noise_model=None,
        plot=False,
        force=True,
    )

    assert isinstance(res, dict)
    assert math.isfinite(float(res["energy"]))
    assert isinstance(res["energies"], list)
    assert isinstance(res["selected_operators"], list)
    assert len(res["selected_operators"]) <= 4


def test_qse_smoke_and_sanity() -> None:
    cfg = dict(
        molecule="H2",
        k=3,
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=6,
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        pool="hamiltonian_topk",
        max_ops=10,
        eps=1e-10,
        force=True,
    )

    res = run_qse(**cfg)

    assert isinstance(res, dict)
    assert "eigenvalues" in res
    assert "diagnostics" in res
    assert "config" in res

    eigs = [float(x) for x in res["eigenvalues"]]
    assert len(eigs) >= 1
    assert all(np.isfinite(eigs))
    assert _is_sorted_non_decreasing(eigs)

    exact = np.asarray(get_exact_spectrum("H2", mapping="jordan_wigner"), dtype=float)
    exact = np.sort(exact)[:20]
    diffs = _nearest_diffs(np.asarray(eigs, dtype=float), exact)

    assert np.max(diffs) < 0.5


def test_lr_vqe_deterministic() -> None:
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


def test_eom_vqe_deterministic() -> None:
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
    assert np.allclose(exc1, exc2, atol=1e-8, rtol=0.0)


def test_eom_qse_smoke_and_deterministic() -> None:
    cfg = dict(
        molecule="H2",
        k=3,
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        steps=20,
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        pool="hamiltonian_topk",
        max_ops=10,
        eps=1e-10,
        imag_tol=1e-10,
        omega_eps=1e-12,
        force=True,
    )

    r1 = run_eom_qse(**cfg)
    r2 = run_eom_qse(**cfg)

    eigs1 = np.asarray(r1["eigenvalues"], dtype=float)
    eigs2 = np.asarray(r2["eigenvalues"], dtype=float)

    assert eigs1.shape == eigs2.shape
    assert eigs1.size >= 1
    assert np.all(np.isfinite(eigs1))
    assert np.allclose(eigs1, eigs2, atol=1e-10, rtol=0.0)
