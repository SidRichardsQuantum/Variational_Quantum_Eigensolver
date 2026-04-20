from __future__ import annotations

import time

import numpy as np

from common import (
    analyze_qpe_result,
    compute_fidelity,
    exact_ground_energy_for_problem,
    qpe_branch_candidates,
    qpe_calibration_plan,
    summarize_problem,
    summary_stats,
    timed_call,
)
from common.persist import cached_compute_runtime


def test_summary_stats_returns_basic_moments() -> None:
    out = summary_stats([1.0, 2.0, 3.0])

    assert out == {
        "mean": 2.0,
        "std": float(np.std(np.asarray([1.0, 2.0, 3.0]), ddof=0)),
        "min": 1.0,
        "max": 3.0,
    }


def test_timed_call_returns_result_and_elapsed() -> None:
    def f(*, x: int) -> int:
        time.sleep(0.01)
        return x + 1

    result, elapsed = timed_call(f, x=2)

    assert result == 3
    assert isinstance(elapsed, float)
    assert elapsed >= 0.0


def test_timed_call_can_suppress_stdout(capsys) -> None:
    def f() -> str:
        print("hidden output")
        return "ok"

    result, _elapsed = timed_call(f, suppress_stdout=True)

    assert result == "ok"
    assert capsys.readouterr().out == ""


def test_compute_fidelity_supports_statevector_and_density_matrix() -> None:
    psi = np.asarray([1.0, 0.0], dtype=complex)
    phi = np.asarray([1.0, 0.0], dtype=complex)
    rho = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=complex)

    assert compute_fidelity(psi, phi) == 1.0
    assert compute_fidelity(psi, rho) == 1.0


def test_compute_fidelity_rejects_invalid_shape() -> None:
    psi = np.asarray([1.0, 0.0], dtype=complex)

    try:
        compute_fidelity(psi, np.asarray([[[1.0]]], dtype=complex))
    except ValueError as exc:
        assert "Invalid state shape" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for invalid state shape")


def test_cached_compute_runtime_prefers_explicit_field_then_legacy_runtime() -> None:
    assert (
        cached_compute_runtime({"compute_runtime_s": 1.25, "runtime_s": 9.99}) == 1.25
    )
    assert cached_compute_runtime({"runtime_s": 2.5}) == 2.5
    assert cached_compute_runtime({"runtime_s": None}) is None
    assert cached_compute_runtime({}) is None


def test_exact_ground_energy_for_problem_matches_summary_problem() -> None:
    spec = {"molecule": "H2", "mapping": "jordan_wigner"}

    exact = exact_ground_energy_for_problem(**spec)
    summary = summarize_problem(**spec)

    assert isinstance(exact, float)
    assert summary["molecule"] == "H2"
    assert summary["mapping"] == "jordan_wigner"
    assert int(summary["num_qubits"]) >= 1
    assert int(summary["hamiltonian_terms"]) >= 1
    assert abs(float(summary["exact_ground_energy"]) - exact) < 1e-12


def test_qpe_branch_candidates_reports_both_orientations() -> None:
    out = qpe_branch_candidates("10", t=1.0, ref_energy=-1.0)

    assert out["phase_msb"] == 0.5
    assert out["phase_lsb"] == 0.25
    assert isinstance(out["energy_msb"], float)
    assert isinstance(out["energy_lsb"], float)
    assert out["energy_msb"] != out["energy_lsb"]


def test_analyze_qpe_result_flags_dominant_bin_failure() -> None:
    result = {
        "t": 1.0,
        "hf_energy": -1.0,
        "energy": 0.0,
        "best_bitstring": "00",
        "probs": {"00": 0.6, "10": 0.4},
        "phase": 0.0,
        "n_ancilla": 2,
    }

    out = analyze_qpe_result(result, exact_ground=-np.pi)

    assert out["best_bitstring"] == "00"
    assert out["oracle_any_bitstring"] == "10"
    assert out["dominant_bin_failure"] is True
    assert out["abs_error"] > out["oracle_any_abs_error"]


def test_analyze_qpe_result_rejects_inconsistent_best_bitstring() -> None:
    result = {
        "t": 1.0,
        "hf_energy": -1.0,
        "energy": 0.0,
        "best_bitstring": "11",
        "probs": {"00": 1.0},
        "phase": 0.0,
        "n_ancilla": 2,
    }

    try:
        analyze_qpe_result(result, exact_ground=-1.0)
    except ValueError as exc:
        assert "best_bitstring" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for inconsistent QPE result")


def test_qpe_calibration_plan_runs_analytic_once_and_sampled_across_seeds() -> None:
    plan = qpe_calibration_plan(
        ancillas_grid=[2],
        times_grid=[1.0],
        trotter_grid=[1],
        shots_grid=[None, 100],
        seeds=[4, 5],
    )

    assert plan == [
        {"ancillas": 2, "t": 1.0, "trotter_steps": 1, "shots": None, "seed": 0},
        {"ancillas": 2, "t": 1.0, "trotter_steps": 1, "shots": 100, "seed": 4},
        {"ancillas": 2, "t": 1.0, "trotter_steps": 1, "shots": 100, "seed": 5},
    ]
