from __future__ import annotations

import numpy as np
import pennylane as qml
import pennylane.qchem as qchem

import common.plotting as common_plotting
import vqe.core as vqe_core
from vqe import run_vqe


def test_vqe_minimal_smoke() -> None:
    res = run_vqe(
        molecule="H2",
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=4,
        noisy=False,
        force=True,
        plot=False,
    )

    assert isinstance(res, dict)
    assert "energy" in res
    assert "energies" in res
    assert "num_qubits" in res
    assert "runtime_s" in res
    assert "compute_runtime_s" in res
    assert "cache_hit" in res

    assert np.isfinite(float(res["energy"]))
    assert len(res["energies"]) >= 1
    assert int(res["num_qubits"]) > 0
    assert float(res["runtime_s"]) >= 0.0
    assert float(res["compute_runtime_s"]) >= 0.0


def test_vqe_deterministic_given_seed_and_force() -> None:
    cfg = dict(
        molecule="H2",
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=4,
        stepsize=0.2,
        seed=0,
        noisy=False,
        force=True,
        plot=False,
    )

    r1 = run_vqe(**cfg)
    r2 = run_vqe(**cfg)

    e1 = np.asarray(r1["energies"], dtype=float)
    e2 = np.asarray(r2["energies"], dtype=float)

    assert e1.shape == e2.shape
    assert np.all(np.isfinite(e1))
    assert np.allclose(e1, e2, atol=1e-10, rtol=0.0)


def test_vqe_prebuilt_hamiltonian_smoke_and_cache_hit() -> None:
    H = qml.Hamiltonian([1.0], [qml.PauliZ(3)])

    cfg = dict(
        molecule="expert_vqe_cache_smoke",
        hamiltonian=H,
        num_qubits=1,
        reference_state=[1],
        ansatz_name="RY-CZ",
        optimizer_name="Adam",
        steps=2,
        stepsize=0.1,
        plot=False,
    )

    fresh = run_vqe(force=True, **cfg)
    res = run_vqe(force=False, **cfg)

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert int(res["num_qubits"]) == 1
    assert fresh["cache_hit"] is False
    assert res["cache_hit"] is True


def test_vqe_routes_ansatz_kwargs_to_non_molecule_ansatz() -> None:
    H = qml.Hamiltonian([1.0], [qml.PauliZ(0)])

    res = run_vqe(
        hamiltonian=H,
        num_qubits=2,
        reference_state=[1, 0],
        ansatz_name="NumberPreservingGivens",
        ansatz_kwargs={"layers": 2},
        optimizer_name="Adam",
        steps=1,
        stepsize=0.1,
        plot=False,
        force=True,
        seed=0,
    )

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert int(res["num_qubits"]) == 2
    assert len(res["final_params"]) == 2


def test_vqe_routes_model_specific_ansatz_kwargs() -> None:
    H = qml.Hamiltonian(
        [-1.0, -0.5],
        [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0)],
    )

    res = run_vqe(
        hamiltonian=H,
        num_qubits=2,
        reference_state=[0, 0],
        ansatz_name="TFIM-HVA",
        ansatz_kwargs={"layers": 2},
        optimizer_name="Adam",
        steps=1,
        stepsize=0.1,
        plot=False,
        force=True,
        seed=0,
    )

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert len(res["final_params"]) == 4


def test_vqe_auto_ansatz_selects_tfim_hva() -> None:
    H = qml.Hamiltonian(
        [-1.0, -0.5, -0.5],
        [
            qml.PauliZ(0) @ qml.PauliZ(1),
            qml.PauliX(0),
            qml.PauliX(1),
        ],
    )

    res = run_vqe(
        hamiltonian=H,
        num_qubits=2,
        reference_state=[0, 0],
        ansatz_name="auto",
        ansatz_kwargs={"layers": 2},
        optimizer_name="Adam",
        steps=1,
        stepsize=0.1,
        plot=False,
        force=True,
        seed=0,
    )

    assert res["ansatz"] == "TFIM-HVA"
    assert res["ansatz_kwargs"] == {"layers": 2}
    assert res["ansatz_selection"]["selected"] == "TFIM-HVA"
    assert len(res["final_params"]) == 4


def test_vqe_cache_hit_reports_cached_timing_metadata() -> None:
    cfg = dict(
        molecule="H2",
        ansatz_name="Minimal",
        optimizer_name="Adam",
        steps=1,
        stepsize=0.2,
        noisy=False,
        plot=False,
        seed=123,
    )

    fresh = run_vqe(force=True, **cfg)
    cached = run_vqe(force=False, **cfg)

    assert fresh["cache_hit"] is False
    assert cached["cache_hit"] is True
    assert float(fresh["compute_runtime_s"]) >= 0.0
    assert float(cached["runtime_s"]) >= 0.0
    assert np.isclose(
        float(cached["compute_runtime_s"]),
        float(fresh["compute_runtime_s"]),
    )


def test_vqe_expert_mode_uccsd_supports_open_shell_multiplicity() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]], dtype=float)
    hamiltonian, num_qubits = qchem.molecular_hamiltonian(
        ["H", "H"],
        coords,
        charge=1,
        mult=2,
        basis="sto-3g",
        unit="angstrom",
        method="openfermion",
        mapping="jordan_wigner",
    )
    mol = qchem.Molecule(
        ["H", "H"],
        coords,
        charge=1,
        mult=2,
        basis_name="sto-3g",
        unit="angstrom",
    )
    hf_state = qchem.hf_state(int(mol.n_electrons), 2 * int(mol.n_orbitals))

    res = run_vqe(
        hamiltonian=hamiltonian,
        num_qubits=int(num_qubits),
        reference_state=hf_state.tolist(),
        symbols=["H", "H"],
        coordinates=coords,
        charge=1,
        multiplicity=2,
        basis="sto-3g",
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        steps=1,
        plot=False,
        force=True,
        seed=0,
    )

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))


def test_vqe_low_qubit_benchmark_aggregates_selected_molecules(monkeypatch) -> None:
    def fake_select_low_qubit_molecules(*, molecules, max_qubits, mapping, unit):
        assert molecules is None
        assert int(max_qubits) == 10
        assert mapping == "jordan_wigner"
        assert unit == "angstrom"
        return [
            {
                "molecule": "H2",
                "num_qubits": 4,
                "exact_ground_energy": -1.10,
                "hamiltonian_terms": 15,
            },
            {
                "molecule": "H3+",
                "num_qubits": 6,
                "exact_ground_energy": -1.90,
                "hamiltonian_terms": 31,
            },
        ]

    def fake_run_vqe(*args, **kwargs):
        mol = kwargs["molecule"]
        seed = int(kwargs["seed"])
        base = {"H2": -1.10, "H3+": -1.90}[mol]
        offset = 0.01 if seed else 0.0
        return {
            "energy": base + offset,
            "final_params": [0.1, 0.2, 0.3],
        }

    def fake_save_plot(filename, *, kind, molecule=None, show=True):
        assert kind == "vqe"
        assert molecule == "multi_molecule"
        assert filename.endswith(".png")
        return f"/tmp/{filename}"

    monkeypatch.setattr(
        vqe_core,
        "_select_low_qubit_molecules",
        fake_select_low_qubit_molecules,
    )
    monkeypatch.setattr(vqe_core, "run_vqe", fake_run_vqe)
    monkeypatch.setattr(common_plotting, "save_plot", fake_save_plot)

    out = vqe_core.run_vqe_low_qubit_benchmark(
        max_qubits=10,
        seeds=[0, 1],
        steps=4,
        show=False,
        force=True,
    )

    assert out["max_qubits"] == 10
    assert out["mapping"] == "jordan_wigner"
    assert out["plot_path"].startswith("/tmp/low_qubit_benchmark")
    assert len(out["rows"]) == 2

    h2 = out["rows"][0]
    h3p = out["rows"][1]

    assert h2["molecule"] == "H2"
    assert h2["num_qubits"] == 4
    assert np.isclose(float(h2["energy_mean"]), -1.095)
    assert np.isclose(float(h2["abs_error_mean"]), 0.005)
    assert int(h2["parameter_count"]) == 3

    assert h3p["molecule"] == "H3+"
    assert h3p["num_qubits"] == 6
    assert np.isclose(float(h3p["energy_mean"]), -1.895)
    assert np.isclose(float(h3p["abs_error_mean"]), 0.005)


def test_vqe_low_qubit_benchmark_skips_failed_molecules(monkeypatch) -> None:
    def fake_select_low_qubit_molecules(*, molecules, max_qubits, mapping, unit):
        return [
            {
                "molecule": "H2",
                "num_qubits": 4,
                "exact_ground_energy": -1.10,
                "hamiltonian_terms": 15,
            },
            {
                "molecule": "He2",
                "num_qubits": 4,
                "exact_ground_energy": -5.60,
                "hamiltonian_terms": 9,
            },
        ]

    def fake_run_vqe(*args, **kwargs):
        if kwargs["molecule"] == "He2":
            raise ValueError("no valid excitations")
        return {"energy": -1.10, "final_params": [0.1]}

    def fake_save_plot(filename, *, kind, molecule=None, show=True):
        return f"/tmp/{filename}"

    monkeypatch.setattr(
        vqe_core,
        "_select_low_qubit_molecules",
        fake_select_low_qubit_molecules,
    )
    monkeypatch.setattr(vqe_core, "run_vqe", fake_run_vqe)
    monkeypatch.setattr(common_plotting, "save_plot", fake_save_plot)

    out = vqe_core.run_vqe_low_qubit_benchmark(show=False)

    assert [row["molecule"] for row in out["rows"]] == ["H2"]
    assert len(out["skipped"]) == 1
    assert out["skipped"][0]["molecule"] == "He2"
    assert "ValueError" in out["skipped"][0]["reason"]
