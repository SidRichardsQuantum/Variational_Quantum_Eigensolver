from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pennylane as qml
import pytest
from pennylane import numpy as pnp

from common.hamiltonian import get_exact_spectrum
import vqe.__main__ as vqe_main
from vqe import run_qse
from vqe import run_ssvqe
from vqe import run_vqd
from vqe.adapt import run_adapt_vqe
from vqe.eom_qse import run_eom_qse
from vqe.eom_vqe import run_eom_vqe
from vqe.lr_vqe import run_lr_vqe


def _is_sorted_non_decreasing(xs: list[float], tol: float = 1e-12) -> bool:
    return all(xs[i] <= xs[i + 1] + tol for i in range(len(xs) - 1))


def _nearest_diffs(values: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    out = []
    for v in values:
        out.append(float(np.min(np.abs(candidates - v))))
    return np.asarray(out, dtype=float)


def test_adapt_vqe_smoke() -> None:
    res = run_adapt_vqe(
        molecule="H2",
        pool="uccs",
        max_ops=4,
        grad_tol=1e-3,
        inner_steps=8,
        inner_stepsize=0.2,
        optimizer_name="Adam",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        noise_model=None,
        plot=False,
        force=True,
    )

    assert isinstance(res, dict)
    assert math.isfinite(float(res["energy"]))
    assert isinstance(res["energies"], list)
    assert isinstance(res["selected_operators"], list)
    assert len(res["selected_operators"]) <= 4


def test_qse_smoke_and_sanity() -> None:
    cfg = dict(
        molecule="H2",
        k=3,
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=6,
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        pool="hamiltonian_topk",
        max_ops=10,
        eps=1e-10,
        force=True,
    )

    res = run_qse(**cfg)

    assert isinstance(res, dict)
    assert "eigenvalues" in res
    assert "diagnostics" in res
    assert "config" in res

    eigs = [float(x) for x in res["eigenvalues"]]
    assert len(eigs) >= 1
    assert all(np.isfinite(eigs))
    assert _is_sorted_non_decreasing(eigs)

    exact = np.asarray(get_exact_spectrum("H2", mapping="jordan_wigner"), dtype=float)
    exact = np.sort(exact)[:20]
    diffs = _nearest_diffs(np.asarray(eigs, dtype=float), exact)

    assert np.max(diffs) < 0.5


def test_lr_vqe_deterministic() -> None:
    cfg = dict(
        molecule="H2",
        k=2,
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=8,
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        fd_eps=1e-3,
        eps=1e-10,
        force=True,
    )

    r1 = run_lr_vqe(**cfg)
    r2 = run_lr_vqe(**cfg)

    exc1 = np.asarray(r1["excitations"], dtype=float)
    exc2 = np.asarray(r2["excitations"], dtype=float)

    assert exc1.shape == exc2.shape
    assert np.all(np.isfinite(exc1))
    assert np.allclose(exc1, exc2, atol=1e-8, rtol=0.0)


def test_eom_vqe_deterministic() -> None:
    cfg = dict(
        molecule="H2",
        k=2,
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        steps=25,
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        fd_eps=1e-3,
        eps=1e-10,
        omega_eps=1e-12,
        force=True,
        plot=False,
        show=False,
        save=False,
    )

    r1 = run_eom_vqe(**cfg)
    r2 = run_eom_vqe(**cfg)

    exc1 = np.asarray(r1["excitations"], dtype=float)
    exc2 = np.asarray(r2["excitations"], dtype=float)

    assert exc1.shape == exc2.shape
    assert exc1.size >= 1
    assert np.all(np.isfinite(exc1))
    assert np.all(exc1 > 0.0)
    assert np.allclose(exc1, exc2, atol=1e-8, rtol=0.0)


def test_eom_qse_smoke_and_deterministic() -> None:
    cfg = dict(
        molecule="H2",
        k=3,
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        steps=20,
        stepsize=0.2,
        seed=0,
        mapping="jordan_wigner",
        pool="hamiltonian_topk",
        max_ops=10,
        eps=1e-10,
        imag_tol=1e-10,
        omega_eps=1e-12,
        force=True,
    )

    r1 = run_eom_qse(**cfg)
    r2 = run_eom_qse(**cfg)

    eigs1 = np.asarray(r1["eigenvalues"], dtype=float)
    eigs2 = np.asarray(r2["eigenvalues"], dtype=float)

    assert eigs1.shape == eigs2.shape
    assert eigs1.size >= 1
    assert np.all(np.isfinite(eigs1))
    assert np.allclose(eigs1, eigs2, atol=1e-10, rtol=0.0)


def test_ssvqe_propagates_mapping_to_hamiltonian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_build_hamiltonian(molecule, mapping="jordan_wigner", unit="angstrom"):
        captured["mapping"] = mapping
        return (
            qml.Hamiltonian([0.0], [qml.Identity(0)]),
            1,
            np.array([1], dtype=int),
            ["H"],
            np.array([[0.0, 0.0, 0.0]], dtype=float),
            "sto-3g",
            0,
            "angstrom",
        )

    def fake_build_ansatz(*args, **kwargs):
        def ansatz_fn(params, wires):
            return None

        return ansatz_fn, pnp.array([], requires_grad=True)

    monkeypatch.setattr("vqe.ssvqe.build_hamiltonian", fake_build_hamiltonian)
    monkeypatch.setattr("vqe.ssvqe.build_ansatz", fake_build_ansatz)

    res = run_ssvqe(
        molecule="H2",
        num_states=2,
        ansatz_name="Minimal",
        steps=1,
        plot=False,
        force=True,
        mapping="parity",
    )

    assert captured["mapping"] == "parity"
    assert res["config"]["mapping"] == "parity"


def test_ssvqe_explicit_geometry_uses_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_hamiltonian(
        molecule=None,
        coordinates=None,
        symbols=None,
        *,
        charge=None,
        basis=None,
        mapping="jordan_wigner",
        unit="angstrom",
    ):
        captured["molecule"] = molecule
        captured["symbols"] = list(symbols)
        captured["coordinates"] = np.array(coordinates, dtype=float)
        captured["charge"] = charge
        captured["basis"] = basis
        captured["mapping"] = mapping
        captured["unit"] = unit
        return (
            qml.Hamiltonian([0.0], [qml.Identity(0)]),
            1,
            np.array([1], dtype=int),
            list(symbols),
            np.array(coordinates, dtype=float),
            "sto-3g",
            int(charge),
            str(unit),
        )

    def fake_build_ansatz(*args, **kwargs):
        def ansatz_fn(params, wires):
            return None

        return ansatz_fn, pnp.array([], requires_grad=True)

    monkeypatch.setattr("vqe.ssvqe.build_hamiltonian", fake_build_hamiltonian)
    monkeypatch.setattr("vqe.ssvqe.build_ansatz", fake_build_ansatz)

    run_ssvqe(
        molecule="custom",
        num_states=2,
        ansatz_name="Minimal",
        steps=1,
        plot=False,
        force=True,
        symbols=["H"],
        coordinates=[[0.0, 0.0, 0.0]],
        basis="6-31g",
        charge=1,
        unit="bohr",
        mapping="parity",
    )

    assert captured["molecule"] is None
    assert captured["symbols"] == ["H"]
    assert np.array_equal(
        captured["coordinates"], np.array([[0.0, 0.0, 0.0]], dtype=float)
    )
    assert captured["charge"] == 1
    assert captured["basis"] == "6-31g"
    assert captured["mapping"] == "parity"
    assert captured["unit"] == "bohr"


def test_vqd_explicit_geometry_uses_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_hamiltonian(
        molecule=None,
        coordinates=None,
        symbols=None,
        *,
        charge=None,
        basis=None,
        mapping="jordan_wigner",
        unit="angstrom",
    ):
        captured["molecule"] = molecule
        captured["symbols"] = list(symbols)
        captured["coordinates"] = np.array(coordinates, dtype=float)
        captured["charge"] = charge
        captured["basis"] = basis
        captured["mapping"] = mapping
        captured["unit"] = unit
        return (
            qml.Hamiltonian([0.0], [qml.Identity(0)]),
            1,
            np.array([1], dtype=int),
            list(symbols),
            np.array(coordinates, dtype=float),
            "sto-3g",
            int(charge),
            str(unit),
        )

    def fake_build_ansatz(*args, **kwargs):
        def ansatz_fn(params, wires):
            return None

        return ansatz_fn, pnp.array([], requires_grad=True)

    monkeypatch.setattr("vqe.vqd.build_hamiltonian", fake_build_hamiltonian)
    monkeypatch.setattr("vqe.vqd.build_ansatz", fake_build_ansatz)

    run_vqd(
        molecule="custom",
        num_states=2,
        ansatz_name="Minimal",
        steps=1,
        plot=False,
        force=True,
        symbols=["H"],
        coordinates=[[0.0, 0.0, 0.0]],
        basis="6-31g",
        charge=1,
        unit="bohr",
        mapping="parity",
    )

    assert captured["molecule"] is None
    assert captured["symbols"] == ["H"]
    assert np.array_equal(
        captured["coordinates"], np.array([[0.0, 0.0, 0.0]], dtype=float)
    )
    assert captured["charge"] == 1
    assert captured["basis"] == "6-31g"
    assert captured["mapping"] == "parity"
    assert captured["unit"] == "bohr"


def test_excited_state_cli_forwards_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    ssvqe_called: dict[str, object] = {}
    vqd_called: dict[str, object] = {}

    def fake_run_ssvqe(**kwargs):
        ssvqe_called.update(kwargs)
        return {"energies_per_state": [[-1.0], [-0.5]]}

    def fake_run_vqd(**kwargs):
        vqd_called.update(kwargs)
        return {"energies_per_state": [[-1.0], [-0.5]]}

    monkeypatch.setattr(vqe_main, "run_ssvqe", fake_run_ssvqe)
    monkeypatch.setattr(vqe_main, "run_vqd", fake_run_vqd)

    ssvqe_args = SimpleNamespace(
        ssvqe=True,
        lr_vqe=False,
        eom_vqe=False,
        eom_qse=False,
        vqd=False,
        noisy=False,
        mapping="parity",
        weights=None,
        num_states=2,
        molecule="H2",
        ansatz="Minimal",
        optimizer="Adam",
        steps=1,
        stepsize=0.1,
        seed=0,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        plot=False,
        force=True,
        basis="6-31g",
        charge=1,
        unit="bohr",
        symbols="H",
        coordinates="0,0,0",
    )

    handled = vqe_main.handle_special_modes(ssvqe_args)
    assert handled is True
    assert ssvqe_called["mapping"] == "parity"
    assert ssvqe_called["symbols"] == ["H"]
    assert np.array_equal(
        ssvqe_called["coordinates"], np.array([[0.0, 0.0, 0.0]], dtype=float)
    )
    assert ssvqe_called["basis"] == "6-31g"
    assert ssvqe_called["charge"] == 1
    assert ssvqe_called["unit"] == "bohr"

    vqd_args = SimpleNamespace(
        ssvqe=False,
        lr_vqe=False,
        eom_vqe=False,
        eom_qse=False,
        vqd=True,
        noisy=False,
        mapping="bravyi_kitaev",
        molecule="H2",
        num_states=2,
        beta=10.0,
        beta_start=None,
        beta_ramp="linear",
        beta_hold_fraction=0.0,
        ansatz="Minimal",
        optimizer="Adam",
        steps=1,
        stepsize=0.1,
        seed=0,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        plot=False,
        force=True,
        basis="6-31g",
        charge=1,
        unit="bohr",
        symbols="H",
        coordinates="0,0,0",
    )

    handled = vqe_main.handle_special_modes(vqd_args)
    assert handled is True
    assert vqd_called["mapping"] == "bravyi_kitaev"
    assert vqd_called["symbols"] == ["H"]
    assert np.array_equal(
        vqd_called["coordinates"], np.array([[0.0, 0.0, 0.0]], dtype=float)
    )
    assert vqd_called["basis"] == "6-31g"
    assert vqd_called["charge"] == 1
    assert vqd_called["unit"] == "bohr"
