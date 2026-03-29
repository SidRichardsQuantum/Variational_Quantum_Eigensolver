from __future__ import annotations

import subprocess
import sys


def _run_help(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_vqe_cli_help() -> None:
    p = _run_help("vqe")
    assert p.returncode == 0
    out = (p.stdout or "") + (p.stderr or "")
    assert "usage" in out.lower()


def test_qpe_cli_help() -> None:
    p = _run_help("qpe")
    assert p.returncode == 0
    out = (p.stdout or "") + (p.stderr or "")
    assert "usage" in out.lower()


def test_qite_cli_help() -> None:
    p = _run_help("qite")
    assert p.returncode == 0
    out = (p.stdout or "") + (p.stderr or "")
    assert "usage" in out.lower()
