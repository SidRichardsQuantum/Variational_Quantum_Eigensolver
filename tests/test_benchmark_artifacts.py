from __future__ import annotations

import os
import re
import subprocess
import sys
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "notebooks" / "benchmarks"
RESULTS = BENCHMARKS / "RESULTS.md"
ARTIFACTS = BENCHMARKS / "_artifacts"
MANIFEST = ARTIFACTS / "benchmark_manifest.json"

EXPECTED_TABLE_COLUMNS = {
    "h2_cross_method_runtime.csv": {"method", "elapsed_s"},
    "h2_noise_reference.csv": {
        "seed",
        "reference_energy",
        "abs_error_to_exact",
        "cache_hit",
    },
    "lih_cross_method_results.csv": {
        "method",
        "energy",
        "exact_ground",
        "abs_error",
        "runtime_s",
        "compute_runtime_s",
        "cache_hit",
        "num_qubits",
    },
    "lih_problem_summary.csv": {"setting", "value"},
    "low_qubit_vqe_summary.csv": {
        "molecule",
        "num_qubits",
        "hamiltonian_terms",
        "exact_ground_energy",
        "energy_mean",
        "abs_error_mean",
        "runtime_mean_s",
    },
    "qpe_h2_best_configurations.csv": {
        "ancillas",
        "t",
        "trotter_steps",
        "shots",
        "seed",
        "energy",
        "abs_error",
        "cache_hit",
    },
    "qpe_h2_ranked_summary.csv": {
        "ancillas",
        "t",
        "trotter_steps",
        "shots_label",
        "mean_abs_error",
        "std_abs_error",
        "max_abs_error",
        "score",
    },
}


def _results_text() -> str:
    return RESULTS.read_text(encoding="utf-8")


def _referenced_images() -> list[Path]:
    return [
        BENCHMARKS / match
        for match in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", _results_text())
        if match.startswith("_artifacts/")
    ]


def _referenced_csvs() -> list[Path]:
    return [
        BENCHMARKS / match
        for match in re.findall(r"CSV artifact: `([^`]+)`", _results_text())
        if match.startswith("_artifacts/")
    ]


def test_results_referenced_artifacts_exist() -> None:
    references = _referenced_images() + _referenced_csvs()
    assert references, "RESULTS.md should reference curated benchmark artifacts"

    missing = [path for path in references if not path.exists()]
    assert not missing


def test_curated_artifacts_are_not_oversized() -> None:
    assert ARTIFACTS.exists()

    max_bytes = 2 * 1024 * 1024
    oversized = [
        path
        for path in ARTIFACTS.rglob("*")
        if path.is_file() and path.stat().st_size > max_bytes
    ]
    assert not oversized


def test_curated_artifact_types_are_expected() -> None:
    allowed_suffixes = {".csv", ".json", ".md", ".png"}
    unexpected = [
        path
        for path in ARTIFACTS.rglob("*")
        if path.is_file() and path.suffix.lower() not in allowed_suffixes
    ]
    assert not unexpected


def test_benchmark_manifest_points_to_existing_artifacts() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1

    referenced: set[Path] = set()
    for figure in manifest["figures"]:
        referenced.add(BENCHMARKS / figure["path"])
        assert figure["title"]
        assert figure["notebook"].endswith(".ipynb")
    for table in manifest["tables"]:
        referenced.add(BENCHMARKS / table["csv"])
        referenced.add(BENCHMARKS / table["md"])
        assert table["title"]
        assert table["notebook"].endswith(".ipynb")

    missing = [path for path in sorted(referenced) if not path.exists()]
    assert not missing

    curated = {
        path
        for path in ARTIFACTS.rglob("*")
        if path.is_file() and path.name != MANIFEST.name
    }
    assert curated == referenced


def test_curated_csv_artifacts_follow_expected_schemas() -> None:
    for filename, expected_columns in EXPECTED_TABLE_COLUMNS.items():
        path = ARTIFACTS / "tables" / filename
        assert path.exists()
        df = pd.read_csv(path)
        missing = expected_columns - set(df.columns)
        assert not missing, f"{filename} missing columns: {sorted(missing)}"
        assert len(df) > 0


def test_exporter_is_idempotent() -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")

    subprocess.run(
        [sys.executable, "scripts/export_benchmark_artifacts.py"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    result = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "notebooks/benchmarks/RESULTS.md",
            "notebooks/benchmarks/_artifacts",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
