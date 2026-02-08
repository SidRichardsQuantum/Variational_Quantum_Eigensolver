# tests/test_qse_minimal.py

from __future__ import annotations

import numpy as np

from vqe import get_exact_spectrum, run_qse


def _is_sorted_non_decreasing(xs: list[float], *, tol: float = 1e-12) -> bool:
    return all(xs[i] <= xs[i + 1] + tol for i in range(len(xs) - 1))


def _nearest_diffs(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    For each value, return min_j |value - candidates[j]|.
    """
    diffs = []
    for v in values:
        diffs.append(float(np.min(np.abs(candidates - v))))
    return np.asarray(diffs, dtype=float)


def test_qse_h2_smoke_and_sanity():
    """
    QSE should:
      - run end-to-end on H2,
      - return k sorted finite eigenvalues,
      - produce reasonable energies (not necessarily chemically converged),
      - include sensible diagnostics + config structure.
    """
    cfg = dict(
        molecule="H2",
        k=3,
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=6,  # keep fast
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        pool="hamiltonian_topk",
        max_ops=10,  # small subspace for test speed
        eps=1e-10,
        force=True,  # avoid cache coupling between test runs
    )

    res = run_qse(**cfg)

    # ---- basic shape / keys ----
    assert isinstance(res, dict)
    assert "eigenvalues" in res
    assert "diagnostics" in res
    assert "config" in res
    assert "num_qubits" in res

    eigs = [float(x) for x in res["eigenvalues"]]
    assert all(np.isfinite(eigs))
    assert _is_sorted_non_decreasing(eigs)

    diag = dict(res["diagnostics"])
    kept_rank = int(diag.get("kept_rank"))
    k_req = int(cfg["k"])

    # QSE can only return up to kept_rank eigenvalues after S filtering.
    assert 1 <= len(eigs) <= k_req
    assert len(eigs) == min(k_req, kept_rank)

    diag = dict(res["diagnostics"])
    assert int(diag.get("subspace_dim")) >= 1
    assert int(diag.get("kept_rank")) >= 1
    assert int(diag.get("kept_rank")) <= int(diag.get("subspace_dim"))
    assert float(diag.get("eps")) == float(cfg["eps"])

    # ---- config sanity (pool / ops recorded) ----
    c = dict(res["config"])
    assert c.get("qse_pool") == "hamiltonian_topk"
    assert int(c.get("qse_k")) == int(cfg["k"])
    assert int(c.get("qse_max_ops")) == int(cfg["max_ops"])
    assert float(c.get("qse_eps")) == float(cfg["eps"])
    assert isinstance(c.get("qse_ops"), list) and len(c["qse_ops"]) >= 1

    # identity should be present as the first operator (as implemented)
    num_qubits = int(res["num_qubits"])
    op0 = dict(c["qse_ops"][0])
    assert op0.get("type") == "pauli_word"
    assert op0.get("word") == ("I" * num_qubits)
    assert op0.get("wires") == list(range(num_qubits))

    # ---- weak physics sanity: QSE eigenvalues should be near *some* exact eigenvalues ----
    # We don't require chemical convergence (few VQE steps), just "not nonsense".
    exact = np.asarray(
        get_exact_spectrum("H2", mapping=str(cfg["mapping"])), dtype=float
    )
    exact = np.sort(exact)[:20]  # enough candidates for matching

    qse_vals = np.asarray(eigs, dtype=float)
    diffs = _nearest_diffs(qse_vals, exact)

    # Loose but meaningful: within 0.5 Ha of the nearest exact level.
    # (Tighten later if you want longer VQE steps in CI.)
    assert np.max(diffs) < 0.5


def test_qse_deterministic_given_seed_and_force():
    """
    With a fixed seed and force=True, two runs should match closely.
    (Important for reproducible caching/hashing assumptions.)
    """
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

    r1 = run_qse(**cfg)
    r2 = run_qse(**cfg)

    e1 = np.asarray(r1["eigenvalues"], dtype=float)
    e2 = np.asarray(r2["eigenvalues"], dtype=float)

    assert e1.shape == e2.shape
    assert np.all(np.isfinite(e1))
    assert np.all(np.isfinite(e2))

    # QSE + VQE reference should be deterministic here.
    assert np.allclose(e1, e2, rtol=0.0, atol=1e-10)
