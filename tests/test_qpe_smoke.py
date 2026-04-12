from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import qpe.io_utils as qpe_io_utils
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


def test_qpe_hamiltonian_override_bypasses_cache_and_plotting(monkeypatch) -> None:
    atoms = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])

    hamiltonian, _, hf_state = build_hamiltonian(
        atoms,
        coords,
        charge=0,
        basis="sto-3g",
    )

    def fail_load(**_kwargs):
        raise AssertionError("expert-mode QPE should not read from cache")

    def fail_save(_result):
        raise AssertionError("expert-mode QPE should not save to cache")

    monkeypatch.setattr(qpe_io_utils, "load_qpe_result", fail_load)
    monkeypatch.setattr(qpe_io_utils, "save_qpe_result", fail_save)

    res = run_qpe(
        hamiltonian=hamiltonian,
        hf_state=hf_state,
        n_ancilla=1,
        shots=100,
        plot=True,
        force=False,
    )

    assert isinstance(res, dict)
    assert "phase" in res


def test_qpe_distribution_displays_right_to_left_kets(monkeypatch) -> None:
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
