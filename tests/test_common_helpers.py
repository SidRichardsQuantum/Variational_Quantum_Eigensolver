from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    analyze_qpe_result,
    compare_benchmark_runs,
    compute_fidelity,
    exact_ground_energy_for_problem,
    ionization_energy_panel,
    list_benchmark_suites,
    qpe_branch_candidates,
    qpe_calibration_plan,
    run_benchmark_suite,
    summarize_problem,
    summary_stats,
    timed_call,
)
from common.persist import cached_compute_runtime

ROOT = Path(__file__).resolve().parents[1]


def test_summary_stats_returns_basic_moments() -> None:
    out = summary_stats([1.0, 2.0, 3.0])

    assert out == {
        "mean": 2.0,
        "std": float(np.std(np.asarray([1.0, 2.0, 3.0]), ddof=0)),
        "min": 1.0,
        "max": 3.0,
    }


def test_ionization_energy_panel_pairs_neutral_and_cation_rows() -> None:
    coverage_rows = [
        {
            "molecule": "He",
            "charge": 0,
            "exact_ground_energy": -2.80,
            "num_electrons": 2,
            "num_qubits": 4,
            "hamiltonian_terms": 9,
        },
        {
            "molecule": "He+",
            "charge": 1,
            "exact_ground_energy": -1.99,
            "num_electrons": 1,
            "num_qubits": 4,
            "hamiltonian_terms": 5,
        },
        {
            "molecule": "Li",
            "charge": 0,
            "exact_ground_energy": -7.43,
            "num_electrons": 3,
            "num_qubits": 10,
            "hamiltonian_terms": 118,
        },
    ]

    rows = ionization_energy_panel(coverage_rows)

    assert len(rows) == 1
    assert rows[0]["system"] == "He"
    assert rows[0]["neutral"] == "He"
    assert rows[0]["cation"] == "He+"
    assert np.isclose(float(rows[0]["ionization_energy_ha"]), 0.81)
    assert float(rows[0]["ionization_energy_ev"]) > 20.0
    assert rows[0]["neutral_num_electrons"] == 2
    assert rows[0]["cation_num_electrons"] == 1


def test_ionization_energy_panel_rejects_missing_requested_pair() -> None:
    coverage_rows = [
        {
            "molecule": "He",
            "charge": 0,
            "exact_ground_energy": -2.80,
            "num_electrons": 2,
            "num_qubits": 4,
            "hamiltonian_terms": 9,
        }
    ]

    try:
        ionization_energy_panel(coverage_rows, pairs=[("He", "He+")])
    except ValueError as exc:
        assert "He+" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for missing cation row")


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


def test_benchmark_suite_registry_includes_fast_expert_suite() -> None:
    suites = list_benchmark_suites()
    ids = {suite["id"] for suite in suites}

    assert "expert-z-cross-method" in ids
    assert "h2-cross-method" in ids
    assert all(suite["title"] for suite in suites)


def test_run_benchmark_suite_writes_research_artifacts(tmp_path) -> None:
    result = run_benchmark_suite(
        "expert-z-cross-method",
        out_dir=tmp_path,
        force=True,
        suppress_stdout=True,
    )

    artifact_paths = {
        key: tmp_path / "expert-z-cross-method" / name
        for key, name in {
            "json": "results.json",
            "csv": "results.csv",
            "markdown": "report.md",
            "manifest": "manifest.json",
        }.items()
    }
    for path in artifact_paths.values():
        assert path.exists()

    payload = json.loads(artifact_paths["json"].read_text(encoding="utf-8"))
    manifest = json.loads(artifact_paths["manifest"].read_text(encoding="utf-8"))

    assert result["suite"]["id"] == "expert-z-cross-method"
    assert payload["suite"]["id"] == "expert-z-cross-method"
    assert payload["schema_version"] == 1
    assert payload["environment"]["python"]
    assert len(payload["rows"]) == 3
    assert {row["method"] for row in payload["rows"]} == {"VQE", "VarQITE", "QPE"}
    assert manifest["row_count"] == 3
    assert manifest["artifacts"]["csv"] == "results.csv"

    with artifact_paths["csv"].open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {"benchmark_id", "method", "energy", "abs_error", "cache_hit"} <= set(
        rows[0]
    )


def test_benchmark_cli_lists_and_runs_suite(tmp_path) -> None:
    listed = subprocess.run(
        [sys.executable, "-m", "common.benchmarks", "list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "expert-z-cross-method" in listed.stdout

    out_dir = tmp_path / "cli"
    ran = subprocess.run(
        [
            sys.executable,
            "-m",
            "common.benchmarks",
            "run",
            "--suite",
            "expert-z-cross-method",
            "--out",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Ran suite: expert-z-cross-method" in ran.stdout
    assert (out_dir / "expert-z-cross-method" / "results.csv").exists()


def test_compare_benchmark_runs_detects_metric_drift(tmp_path) -> None:
    run_benchmark_suite(
        "expert-z-cross-method",
        out_dir=tmp_path / "base",
        force=True,
        suppress_stdout=True,
    )
    base_dir = tmp_path / "base" / "expert-z-cross-method"
    head_dir = tmp_path / "head" / "expert-z-cross-method"
    shutil.copytree(base_dir, head_dir)

    clean = compare_benchmark_runs(base_dir, head_dir)
    assert clean["passed"] is True
    assert clean["changes"] == []

    head_json = head_dir / "results.json"
    payload = json.loads(head_json.read_text(encoding="utf-8"))
    payload["rows"][0]["energy"] = float(payload["rows"][0]["energy"]) + 0.01
    payload["rows"][0]["abs_error"] = float(payload["rows"][0]["abs_error"]) + 0.01
    head_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    drift = compare_benchmark_runs(base_dir, head_dir, energy_tol=1e-6)
    assert drift["passed"] is False
    assert any(change["field"] == "energy" for change in drift["changes"])


def test_benchmark_cli_compare_returns_nonzero_on_drift(tmp_path) -> None:
    run_benchmark_suite(
        "expert-z-cross-method",
        out_dir=tmp_path / "base",
        force=True,
        suppress_stdout=True,
    )
    base_dir = tmp_path / "base" / "expert-z-cross-method"
    head_dir = tmp_path / "head" / "expert-z-cross-method"
    shutil.copytree(base_dir, head_dir)

    head_json = head_dir / "results.json"
    payload = json.loads(head_json.read_text(encoding="utf-8"))
    payload["rows"][0]["compute_runtime_s"] = (
        float(payload["rows"][0]["compute_runtime_s"]) * 10.0 + 1.0
    )
    head_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_path = tmp_path / "compare.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "common.benchmarks",
            "compare",
            "--base",
            str(base_dir),
            "--head",
            str(head_dir),
            "--out",
            str(report_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert any(change["field"] == "compute_runtime_s" for change in report["changes"])
