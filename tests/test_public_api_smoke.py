# tests/test_public_api_smoke.py

from __future__ import annotations

import subprocess
import sys


def test_imports_smoke() -> None:
    import common  # noqa: F401
    import qpe  # noqa: F401
    import qite  # noqa: F401
    import vqe  # noqa: F401

    from qpe.core import run_qpe
    from qite.core import run_qite
    from vqe.core import run_vqe

    assert callable(run_vqe)
    assert callable(run_qpe)
    assert callable(run_qite)


def _run_help(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_cli_help_smoke_vqe() -> None:
    p = _run_help("vqe")
    assert p.returncode == 0
    out = (p.stdout or "") + (p.stderr or "")
    assert "usage" in out.lower()


def test_cli_help_smoke_qpe() -> None:
    p = _run_help("qpe")
    assert p.returncode == 0
    out = (p.stdout or "") + (p.stderr or "")
    assert "usage" in out.lower()


def test_cli_help_smoke_qite() -> None:
    p = _run_help("qite")
    assert p.returncode == 0
    out = (p.stdout or "") + (p.stderr or "")
    assert "usage" in out.lower()
