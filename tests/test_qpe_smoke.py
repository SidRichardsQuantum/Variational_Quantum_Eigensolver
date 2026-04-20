from __future__ import annotations

import subprocess
import sys

import pennylane as qml
import pytest
import numpy as np

from common.hamiltonian import build_hamiltonian
from qpe import run_qpe
from qpe.visualize import plot_qpe_distribution


def test_qpe_minimal_smoke() -> None:
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    hamiltonian, _, hf_state = build_hamiltonian(
        atoms,
        coords,
        charge=0,
        basis="sto-3g",
    )

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


def test_qpe_probability_dict_has_mass() -> None:
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    hamiltonian, _, hf_state = build_hamiltonian(
        atoms,
        coords,
        charge=0,
        basis="sto-3g",
    )

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


def test_qpe_analytic_mode_smoke() -> None:
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    hamiltonian, _, hf_state = build_hamiltonian(
        atoms,
        coords,
        charge=0,
        basis="sto-3g",
    )

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


def test_qpe_hamiltonian_override_uses_cache() -> None:
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    hamiltonian, _, hf_state = build_hamiltonian(
        atoms,
        coords,
        charge=0,
        basis="sto-3g",
    )

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

    monkeypatch.setattr(plt, "close", lambda *args, **kwargs: None)
    plot_qpe_distribution(result, show=False, save=False)
    labels = [tick.get_text() for tick in plt.gca().get_xticklabels()]
    plt.close("all")

    assert labels == ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]


def test_qpe_cli_supports_explicit_geometry() -> None:
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "qpe",
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
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert p.returncode == 0
    assert "QPE completed" in p.stdout
    assert "system=4, ancillas=1" in p.stdout


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
