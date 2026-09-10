"""Reproductions of the v0.3.27 package review findings."""

import numpy as np
import pennylane as qml
import pytest

from common.encoding import occupation_bits
from common.hamiltonian import build_hamiltonian, hartree_fock_state_from_molecule
from common.persist import stable_hash_cfg, stable_hash_dict
from common.problem import resolve_problem
from common.molecules import MOLECULES
from qite import run_qite, run_qrte
from qite import engine as qite_engine
from qpe import run_qpe
from vqe import run_vqe, run_vqd, run_ssvqe
from vqe import run_adapt_vqe, run_eom_vqe, run_lr_vqe
from vqe.engine import build_ansatz, make_state_qnode


@pytest.mark.parametrize(
    "run,options,energy_key",
    [
        (run_adapt_vqe, {"max_ops": 1, "inner_steps": 1}, "energy"),
        (run_vqd, {"steps": 1, "stepsize": 0.0}, "energies_per_state"),
        (run_ssvqe, {"steps": 1, "stepsize": 0.0}, "energies_per_state"),
        (run_eom_vqe, {"steps": 1, "stepsize": 0.0}, "reference_energy"),
        (run_lr_vqe, {"steps": 1, "stepsize": 0.0}, "reference_energy"),
    ],
)
def test_triplet_registry_reference_reaches_other_workflows(
    monkeypatch, run, options, energy_key
):
    monkeypatch.setitem(
        MOLECULES,
        "H3+",
        {
            "symbols": ["H", "H", "H"],
            "coordinates": np.array([[0, 0, 0], [0, 0, 0.8], [0, 0, 1.6]]),
            "charge": 1,
            "multiplicity": 3,
            "basis": "sto-3g",
            "unit": "angstrom",
        },
    )
    H, _, _ = build_hamiltonian("H3+")
    # Highest-weight two-electron reference |101000>.
    expected = qml.matrix(H, wire_order=range(6))[40, 40].real
    result = run(molecule="H3+", plot=False, force=True, **options)
    energy = result[energy_key]
    if energy_key == "energies_per_state":
        energy = energy[0][0]
    if run == run_adapt_vqe:
        energy = result["energies"][0]
    assert energy == pytest.approx(expected, abs=1e-8)


def test_triplet_active_space_retains_unpaired_electrons():
    hf = hartree_fock_state_from_molecule(
        symbols=["H"] * 4,
        coordinates=np.array([[0, 0, z] for z in [0, 0.8, 1.6, 2.4]]),
        charge=0,
        multiplicity=3,
        basis="sto-3g",
        n_qubits=6,
        active_electrons=2,
        active_orbitals=3,
    )
    np.testing.assert_array_equal(hf, [1, 0, 1, 0, 0, 0])


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_triplet_reference_and_ucc_excitation_sector(mapping):
    # Two alpha electrons in three spatial orbitals: a nonempty UCC pool.
    symbols = ["H", "H", "H"]
    coordinates = np.array([[0, 0, 0], [0, 0, 0.8], [0, 0, 1.6]])
    hf = hartree_fock_state_from_molecule(
        symbols=symbols,
        coordinates=coordinates,
        charge=1,
        multiplicity=3,
        basis="sto-3g",
        n_qubits=6,
        mapping=mapping,
    )
    np.testing.assert_array_equal(occupation_bits(hf, mapping), [1, 0, 1, 0, 0, 0])
    ansatz, params = build_ansatz(
        "UCCSD",
        6,
        symbols=symbols,
        coordinates=coordinates,
        charge=1,
        multiplicity=3,
        mapping=mapping,
    )
    assert params.size == 2
    state = make_state_qnode(
        qml.device("default.qubit", wires=6),
        ansatz,
        6,
        symbols=symbols,
        coordinates=coordinates,
        charge=1,
        multiplicity=3,
    )(qml.numpy.array([0.3, -0.7]))
    # Decode every populated determinant independently and check its spin.
    for index, probability in enumerate(abs(state) ** 2):
        if probability > 1e-12:
            bits = occupation_bits([int(c) for c in format(index, "06b")], mapping)
            assert sum(bits[::2]) == 2
            assert sum(bits[1::2]) == 0


@pytest.mark.parametrize("run", [run_vqe, run_qite, run_qrte])
def test_triplet_h2_does_not_collapse_to_singlet(run):
    result = run(
        molecule="triplet_h2",
        symbols=["H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.7414]],
        multiplicity=3,
        ansatz_name="UCCSD",
        steps=1,
        plot=False,
        force=True,
    )
    state = np.array(result["final_state_real"]) + 1j * np.array(
        result["final_state_imag"]
    )
    s2 = qml.matrix(qml.qchem.spin2(2, 4), wire_order=range(4))
    assert np.vdot(state, s2 @ state).real == pytest.approx(2)
    assert result["energy"] > -0.6
    cached = run(
        molecule="triplet_h2",
        symbols=["H", "H"],
        coordinates=[[0, 0, 0], [0, 0, 0.7414]],
        multiplicity=3,
        ansatz_name="UCCSD",
        steps=1,
        plot=False,
    )
    assert cached["cache_hit"] is True
    assert cached["energy"] == result["energy"]


def test_qpe_trotter_order_is_consistent_with_cache_identity():
    first = qml.Hamiltonian([0.8, 1.1], [qml.X(0), qml.Y(0)])
    # Reverse order and split a repeated term: the same operator and cache key.
    second = qml.Hamiltonian([1.1, 0.3, 0.5], [qml.Y(0), qml.X(0), qml.X(0)])
    options = dict(hf_state=[0], n_ancilla=4, shots=None, trotter_steps=1, plot=False)
    original = run_qpe(hamiltonian=first, force=True, **options)
    cached = run_qpe(hamiltonian=second, **options)
    fresh = run_qpe(hamiltonian=second, force=True, **options)
    assert cached["cache_hit"] is True
    assert cached["energy"] == pytest.approx(fresh["energy"])
    assert original["probs"] == pytest.approx(fresh["probs"])


@pytest.mark.parametrize(
    "run",
    [
        resolve_problem,
        build_hamiltonian,
        run_vqe,
        run_qite,
        run_qrte,
        run_qpe,
        run_vqd,
        run_ssvqe,
    ],
)
@pytest.mark.parametrize(
    "geometry", [{"symbols": ["He"]}, {"coordinates": [[0, 0, 0]]}]
)
def test_incomplete_geometry_is_rejected(run, geometry):
    with pytest.raises(ValueError, match="symbols and coordinates.*together"):
        run(**geometry)


@pytest.mark.parametrize("run", [run_qite, run_qrte])
def test_invalid_solver_rejected_even_without_steps(run):
    with pytest.raises(ValueError, match="solver must be"):
        run(solver="bogus", steps=0, plot=False)


@pytest.mark.parametrize(
    "step,time_arg", [(qite_engine.qite_step, "dtau"), (qite_engine.qrte_step, "dt")]
)
def test_low_level_invalid_solver_rejected_before_evaluation(step, time_arg):
    def unexpected(_):
        pytest.fail("Invalid solver must be rejected before evaluating a circuit")

    with pytest.raises(ValueError, match="solver must be"):
        step(
            params=qml.numpy.array([]),
            energy_qnode=unexpected,
            state_qnode=unexpected,
            hamiltonian=qml.Hamiltonian([1], [qml.Z(0)]),
            num_wires=1,
            solver="bogus",
            **{time_arg: 0.1},
        )


@pytest.mark.parametrize("run,key", [(run_qite, "varqite"), (run_qrte, "varqrte")])
@pytest.mark.parametrize("fallback", ["lstsq", "pinv"])
def test_solver_fallback_is_saved_and_reused(monkeypatch, run, key, fallback):
    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError("singular matrix")

    monkeypatch.setattr(qite_engine.np.linalg, "solve", singular)
    if fallback == "pinv":
        monkeypatch.setattr(qite_engine.np.linalg, "lstsq", singular)
    options = dict(
        hamiltonian=qml.Hamiltonian([1], [qml.X(0)]),
        reference_state=[0],
        ansatz_name="RY-CZ",
        solver=" SOLVE ",
        steps=2,
        plot=False,
    )
    fresh = run(force=True, **options)
    cached = run(**options)
    assert fresh[key]["solver"] == "solve"
    assert fresh[key]["solver_history"] == [fallback, fallback]
    assert cached["cache_hit"] is True
    assert cached[key] == fresh[key]


def test_solver_does_not_hide_programming_errors(monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("invalid matrix")

    monkeypatch.setattr(qite_engine.np.linalg, "solve", invalid)
    with pytest.raises(ValueError, match="invalid matrix"):
        run_qite(
            hamiltonian=qml.Hamiltonian([1], [qml.Z(0)]),
            ansatz_name="RY-CZ",
            steps=1,
            plot=False,
            force=True,
        )


def test_previous_cache_schema_is_invalidated():
    cfg = {"molecule": "H2", "multiplicity": 3}
    assert stable_hash_cfg(cfg) != stable_hash_dict({"schema": 2, "config": cfg})
