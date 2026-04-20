from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "notebooks" / "benchmarks"
RESULTS = BENCHMARKS / "RESULTS.md"
ARTIFACTS = BENCHMARKS / "_artifacts"


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
    allowed_suffixes = {".csv", ".md", ".png"}
    unexpected = [
        path
        for path in ARTIFACTS.rglob("*")
        if path.is_file() and path.suffix.lower() not in allowed_suffixes
    ]
    assert not unexpected


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
