from __future__ import annotations

from copy import deepcopy
import subprocess
import sys

import pennylane as qml
import pytest
import numpy as np

from common.hamiltonian import build_hamiltonian
import qpe.__main__ as qpe_main
from qpe import run_qpe
from qpe.visualize import plot_qpe_distribution


@pytest.fixture(scope="module")
def _h2_problem():
    # Chemistry is input setup here; geometry construction has dedicated tests.
    return build_hamiltonian(
        ["H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]),
        charge=0,
        basis="sto-3g",
    )


@pytest.fixture
def h2_problem(_h2_problem):
    # Keep each solver test's mutable inputs independent, as well as its tmp_path
    # result cache. Never share a solver result between tests.
    return deepcopy(_h2_problem)


def test_qpe_minimal_smoke(h2_problem) -> None:
    hamiltonian, _, hf_state = h2_problem

    res = run_qpe(
        hamiltonian=hamiltonian,
        hf_state=hf_state,
        n_ancilla=1,
        shots=200,
    )

    assert isinstance(res, dict)
    assert "phase" in res
    assert "probs" in res
    assert "runtime_s" in res
    assert "compute_runtime_s" in res
    assert "cache_hit" in res
    assert "environment" in res
    assert res["environment"]["python"]
    assert "pennylane" in res["environment"]["packages"]


def test_qpe_probability_dict_has_mass(h2_problem) -> None:
    hamiltonian, _, hf_state = h2_problem

    res = run_qpe(
        hamiltonian=hamiltonian,
        hf_state=hf_state,
        n_ancilla=1,
        shots=200,
    )

    probs = res["probs"]
    assert isinstance(probs, dict)
    assert len(probs) >= 1

    total = sum(float(v) for v in probs.values())
    assert 0.0 < total <= 1.0


def test_qpe_analytic_mode_smoke(h2_problem) -> None:
    hamiltonian, _, hf_state = h2_problem

    res = run_qpe(
        hamiltonian=hamiltonian,
        hf_state=hf_state,
        n_ancilla=1,
        shots=None,
        plot=False,
    )

    assert isinstance(res, dict)
    assert res["shots"] is None
    assert "phase" in res
    assert "0" in res["probs"] or "1" in res["probs"]
    assert abs(sum(float(v) for v in res["probs"].values()) - 1.0) < 1e-9


def test_qpe_explicit_geometry_mode_smoke() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    res = run_qpe(
        symbols=["H", "H"],
        coordinates=coords,
        charge=0,
        basis="sto-3g",
        n_ancilla=1,
        shots=200,
        plot=False,
        force=True,
    )

    assert isinstance(res, dict)
    assert "phase" in res
    assert "probs" in res


def test_qpe_hamiltonian_override_uses_cache(h2_problem) -> None:
    hamiltonian, _, hf_state = h2_problem

    cfg = dict(
        molecule="expert_qpe_cache_smoke",
        hamiltonian=hamiltonian,
        hf_state=hf_state,
        n_ancilla=1,
        shots=100,
        plot=False,
    )

    fresh = run_qpe(force=True, **cfg)
    res = run_qpe(force=False, **cfg)

    assert isinstance(res, dict)
    assert "phase" in res
    assert fresh["cache_hit"] is False
    assert res["cache_hit"] is True
    assert "environment" in fresh
    assert "environment" in res


def test_qpe_cache_hit_reports_cached_timing_metadata() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    cfg = dict(
        symbols=["H", "H"],
        coordinates=coords,
        charge=0,
        basis="sto-3g",
        n_ancilla=1,
        shots=100,
        plot=False,
        seed=222,
    )

    fresh = run_qpe(force=True, **cfg)
    cached = run_qpe(force=False, **cfg)

    assert fresh["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert np.isclose(
        float(cached["compute_runtime_s"]),
        float(fresh["compute_runtime_s"]),
    )


def test_qpe_rejects_partial_expert_mode() -> None:
    H = qml.Hamiltonian([1.0], [qml.PauliZ(0)])

    with pytest.raises(ValueError, match="both hamiltonian and hf_state"):
        run_qpe(hamiltonian=H, n_ancilla=1, shots=10, plot=False)


def test_qpe_distribution_displays_right_to_left_kets(monkeypatch) -> None:
    import matplotlib.pyplot as plt

    result = {
        "molecule": "H2",
        "n_ancilla": 2,
        "probs": {"00": 0.4, "10": 0.3, "01": 0.2, "11": 0.1},
        "noise": {},
        "t": 1.0,
    }

    try:
        with monkeypatch.context() as patch:
            patch.setattr(plt, "close", lambda *args, **kwargs: None)
            plot_qpe_distribution(result, show=False, save=False)
            labels = [tick.get_text() for tick in plt.gca().get_xticklabels()]

        assert labels == ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    finally:
        # Restore the real close function before cleanup, even on assertion failure.
        plt.close("all")


def test_qpe_cli_supports_explicit_geometry(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run_qpe(**kwargs):
        captured.update(kwargs)
        return {
            "best_bitstring": "0",
            "energy": -1.0,
            "hf_energy": -0.9,
            "num_qubits": 4,
        }

    monkeypatch.setattr(qpe_main, "ensure_dirs", lambda: None)
    monkeypatch.setattr(qpe_main, "run_qpe", fake_run_qpe)

    qpe_main.main(
        [
            "--symbols",
            "H,H",
            "--coordinates",
            "0,0,0; 0,0,0.7",
            "--charge",
            "0",
            "--basis",
            "sto-3g",
            "--ancillas",
            "1",
            "--shots",
            "50",
            "--force",
        ]
    )
    out = capsys.readouterr().out

    assert captured["molecule"] == "H2"
    assert captured["symbols"] == ["H", "H"]
    assert np.array_equal(
        captured["coordinates"],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]], dtype=float),
    )
    assert captured["charge"] == 0
    assert captured["basis"] == "sto-3g"
    assert captured["n_ancilla"] == 1
    assert captured["shots"] == 50
    assert captured["force"] is True
    assert "QPE completed" in out
    assert "system=4, ancillas=1" in out


@pytest.mark.slow
@pytest.mark.cli_subprocess
def test_qpe_cli_returns_nonzero_on_failure() -> None:
    p = subprocess.run(
        [sys.executable, "-m", "qpe", "--molecule", "DOES_NOT_EXIST"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    out = (p.stdout or "") + (p.stderr or "")
    assert p.returncode != 0
    assert "Unknown molecule" in out
