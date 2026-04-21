from __future__ import annotations

import subprocess
import sys

import pytest

import qite.__main__ as qite_main
import qpe.__main__ as qpe_main
import vqe.__main__ as vqe_main


def _run_help(module: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "build_parser",
    [vqe_main.build_parser, qpe_main.build_parser, qite_main.build_parser],
)
def test_cli_help_is_available_in_process(build_parser) -> None:
    out = build_parser().format_help()
    assert "usage" in out.lower()


@pytest.mark.slow
@pytest.mark.cli_subprocess
@pytest.mark.parametrize("module", ["vqe", "qpe", "qite"])
def test_module_cli_help(module: str) -> None:
    p = _run_help(module)
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
    vqe_main.main(["--molecule", "H2", "--steps", "1", "--force"])
    out = capsys.readouterr().out

    assert captured["stepsize"] is None
    assert "Stepsize: auto (calibrated per optimizer)" in out
