"""
tests.test_eom_qse_minimal

Smoke + determinism tests for EOM-QSE (operator-manifold commutator EOM).
"""

from __future__ import annotations

import numpy as np


def _is_sorted_non_decreasing(xs: list[float], *, tol: float = 1e-12) -> bool:
    return all(xs[i] <= xs[i + 1] + tol for i in range(len(xs) - 1))


def _nearest_diffs(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    diffs = []
    for v in values:
        diffs.append(float(np.min(np.abs(candidates - v))))
    return np.asarray(diffs, dtype=float)


def test_eom_qse_h2_smoke_and_sanity(tmp_path, monkeypatch):
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))

    from vqe.eom_qse import run_eom_qse
    from vqe import get_exact_spectrum

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

    res = run_eom_qse(**cfg)

    assert isinstance(res, dict)
    for key in [
        "excitations",
        "eigenvalues",
        "diagnostics",
        "config",
        "num_qubits",
        "reference_energy",
    ]:
        assert key in res

    exc = [float(x) for x in res["excitations"]]
    eigs = [float(x) for x in res["eigenvalues"]]

    assert len(exc) >= 1
    assert all(np.isfinite(exc))
    assert all(np.isfinite(eigs))
    assert _is_sorted_non_decreasing(exc)
    assert _is_sorted_non_decreasing(eigs)

    diag = dict(res["diagnostics"])
    kept_rank = int(diag.get("kept_rank"))

    # Internal consistency diagnostics
    assert int(diag.get("num_eigs_total_reduced")) == kept_rank
    assert int(diag.get("num_eigs_positive_realish")) >= len(exc)

    assert 1 <= len(exc) <= int(cfg["k"])
    # Non-Hermitian EOM-QSE filters to positive, real-ish modes; it may return fewer than k.
    assert len(exc) <= min(int(cfg["k"]), kept_rank)

    c = dict(res["config"])
    assert c.get("eom_qse_pool") == "hamiltonian_topk"
    assert int(c.get("eom_qse_k")) == int(cfg["k"])
    assert int(c.get("eom_qse_max_ops")) == int(cfg["max_ops"])
    assert float(c.get("eom_qse_eps")) == float(cfg["eps"])
    assert isinstance(c.get("eom_qse_ops"), list) and len(c["eom_qse_ops"]) >= 1

    # identity present first
    num_qubits = int(res["num_qubits"])
    op0 = dict(c["eom_qse_ops"][0])
    assert op0.get("type") == "pauli_word"
    assert op0.get("word") == ("I" * num_qubits)
    assert op0.get("wires") == list(range(num_qubits))

    # weak physics sanity: E0 + ω near *some* exact eigenvalues (loose)
    exact = np.asarray(
        get_exact_spectrum("H2", mapping=str(cfg["mapping"])), dtype=float
    )
    exact = np.sort(exact)[:20]
    diffs = _nearest_diffs(np.asarray(eigs, dtype=float), exact)
    assert np.median(diffs) < 0.8


def test_eom_qse_deterministic_given_seed_and_force(tmp_path, monkeypatch):
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))

    from vqe.eom_qse import run_eom_qse

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

    e1 = np.asarray(r1["eigenvalues"], dtype=float)
    e2 = np.asarray(r2["eigenvalues"], dtype=float)

    assert e1.shape == e2.shape
    assert np.all(np.isfinite(e1))
    assert np.allclose(e1, e2, rtol=0.0, atol=1e-10)
