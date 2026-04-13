from __future__ import annotations

import subprocess
import sys

import vqe.__main__ as vqe_main


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


def test_vqe_cli_omitted_stepsize_preserves_auto_default(
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_vqe(**kwargs):
        captured.update(kwargs)
        return {"energy": -1.0, "energies": [-1.0], "num_qubits": 1}

    monkeypatch.setattr(vqe_main, "run_vqe", fake_run_vqe)
    monkeypatch.setattr(
        sys,
        "argv",
        ["vqe", "--molecule", "H2", "--steps", "1", "--force"],
    )

    vqe_main.main()
    out = capsys.readouterr().out

    assert captured["stepsize"] is None
    assert "Stepsize: auto (calibrated per optimizer)" in out
