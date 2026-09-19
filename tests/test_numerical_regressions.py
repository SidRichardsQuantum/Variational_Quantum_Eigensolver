"""Numerical contracts for the v0.3.27 correctness fixes."""

import numpy as np
import pennylane as qml
import pytest

from common.encoding import apply_encoding
from common.hamiltonian import (
    build_hamiltonian,
    build_molecular_hamiltonian,
    summarize_registry_coverage,
)
from common.persist import stable_hash_cfg, stable_hash_dict
from common.problem import resolve_problem
from qite import run_qite, run_qrte
from qpe import run_qpe
from qpe.core import inverse_qft, phase_to_energy_unwrapped
from vqe import run_vqe, run_adapt_vqe, run_ssvqe, run_vqd
from vqe.adapt import _inner_optimize


def test_qpe_retains_identity_phase():
    hamiltonian = qml.Hamiltonian([np.pi / 4, np.pi / 2], [qml.Z(0), qml.I(0)])
    result = run_qpe(
        hamiltonian=hamiltonian,
        hf_state=[0],
        n_ancilla=3,
        shots=None,
        plot=False,
        force=True,
    )
    assert result["energy"] == pytest.approx(3 * np.pi / 4)
    assert result["best_bitstring"] == "101"
    assert result["probs"]["101"] == pytest.approx(1)


def test_inverse_qft_respects_wire_labels_and_bit_order():
    wires = [3, 5, 8]
    actual = qml.matrix(inverse_qft, wire_order=wires)(wires)
    expected = qml.matrix(qml.adjoint(qml.QFT)(wires=wires))
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("energy", [-200.0, -20.0, 20.0, 200.0])
def test_qpe_unwraps_arbitrarily_distant_reference(energy):
    t = 1.3
    phase = (-energy * t / (2 * np.pi)) % 1
    assert phase_to_energy_unwrapped(phase, t, energy) == pytest.approx(energy)


@pytest.mark.parametrize("t", [0.0, -1.0, np.inf, np.nan])
def test_qpe_rejects_invalid_evolution_time(t):
    with pytest.raises(ValueError, match="finite and positive"):
        run_qpe(t=t, plot=False)
    with pytest.raises(ValueError, match="finite and positive"):
        phase_to_energy_unwrapped(0.2, t)


@pytest.mark.parametrize(
    "option,channel",
    [
        ("phase_damping_prob", qml.PhaseDamping),
        ("bit_flip_prob", qml.BitFlip),
        ("phase_flip_prob", qml.PhaseFlip),
    ],
)
def test_qpe_noise_matches_explicit_channel_circuit(option, channel):
    # Diagonal H: exact controlled evolution is simple and
    # independent of the package's evolution/noise helper implementation.
    hamiltonian = qml.Hamiltonian([np.pi / 4], [qml.Z(0)])
    dev = qml.device("default.mixed", wires=3)

    @qml.qnode(dev)
    def expected():
        for wire in [0, 1]:
            qml.Hadamard(wire)
        for control in [0, 0, 1]:
            qml.ctrl(qml.RZ, control=control)(np.pi / 2, wires=2)
            channel(0.4, wires=control)
            channel(0.4, wires=2)
        qml.adjoint(qml.QFT)(wires=[0, 1])
        return qml.probs(wires=[0, 1])

    result = run_qpe(
        hamiltonian=hamiltonian,
        hf_state=[0],
        n_ancilla=2,
        shots=None,
        plot=False,
        force=True,
        noisy=True,
        **{option: 0.4},
    )
    np.testing.assert_allclose(
        [result["probs"].get(format(i, "02b"), 0.0) for i in range(4)],
        expected(),
        atol=1e-12,
    )


@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_mapped_hamiltonian_and_hf_are_consistent(mapping):
    original, n, original_hf = build_hamiltonian("H2")
    mapped, _, mapped_hf = build_hamiltonian("H2", mapping=mapping)
    encoding = qml.matrix(apply_encoding, wire_order=range(n))(range(n), mapping)
    original_matrix = qml.matrix(original, wire_order=range(n))
    mapped_matrix = qml.matrix(mapped, wire_order=range(n))
    np.testing.assert_allclose(
        mapped_matrix, encoding @ original_matrix @ encoding.T, atol=1e-12
    )
    state = np.eye(2**n)[:, int("".join(map(str, original_hf)), 2)]
    assert np.argmax(encoding @ state) == int("".join(map(str, mapped_hf)), 2)


@pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
@pytest.mark.parametrize(
    "runner,options",
    [
        (run_vqe, {"optimizer_name": "GradientDescent"}),
        (run_qite, {"show": False}),
        (run_qrte, {"show": False}),
    ],
)
def test_ucc_trajectory_is_equivalent_across_encodings(mapping, runner, options):
    cfg = dict(
        molecule="H2", ansatz_name="UCCSD", steps=2, plot=False, force=True, **options
    )
    original = runner(**cfg)
    mapped = runner(**cfg, mapping=mapping)
    np.testing.assert_allclose(mapped["energies"], original["energies"], atol=1e-9)
    n = mapped["num_qubits"]
    encoding = qml.matrix(apply_encoding, wire_order=range(n))(range(n), mapping)

    def state(r):
        return np.array(r["final_state_real"]) + 1j * np.array(r["final_state_imag"])

    np.testing.assert_allclose(state(mapped), encoding @ state(original), atol=1e-9)


def test_backend_retry_never_drops_requested_mapping(monkeypatch):
    calls = []

    def unsupported(**kwargs):
        calls.append(kwargs)
        raise TypeError("mapping unsupported")

    monkeypatch.setattr(qml.qchem, "molecular_hamiltonian", unsupported)
    with pytest.raises(RuntimeError, match="mapping unsupported"):
        build_molecular_hamiltonian(
            symbols=["H", "H"],
            coordinates=np.array([[0, 0, 0], [0, 0, 0.7]]),
            charge=0,
            basis="sto-3g",
            mapping="parity",
        )
    assert len(calls) == 2
    assert all(call["mapping"] == "parity" for call in calls)


@pytest.mark.parametrize("terms", [[qml.Z(1), qml.Z(0)], [qml.Z(0), qml.Z(1)]])
def test_expert_wire_order_preserves_reference_meaning(terms):
    coefficients = [1.0 if op.wires[0] == 1 else 2.0 for op in terms]
    hamiltonian = qml.Hamiltonian(coefficients, terms)
    problem = resolve_problem(
        hamiltonian=hamiltonian, num_qubits=2, reference_state=[1, 0]
    )
    assert qml.matrix(problem.hamiltonian, wire_order=[0, 1])[2, 2] == pytest.approx(
        -1.0
    )


def test_expert_sparse_integer_wires_respect_explicit_register():
    problem = resolve_problem(
        hamiltonian=qml.Hamiltonian([1.0], [qml.Z(2)]), num_qubits=3
    )
    assert problem.num_qubits == 3
    assert list(problem.hamiltonian.wires) == [2]
    with pytest.raises(ValueError, match="register required"):
        resolve_problem(
            hamiltonian=qml.Hamiltonian([1.0], [qml.Z(0) @ qml.Z(1)]), num_qubits=1
        )


def test_vqe_energy_matches_final_state_and_parameter_history():
    hamiltonian = qml.Hamiltonian([1.0], [qml.Z(0)])
    result = run_vqe(
        hamiltonian=hamiltonian,
        num_qubits=1,
        reference_state=[0],
        ansatz_name="RY-CZ",
        steps=2,
        stepsize=0.5,
        plot=False,
        force=True,
    )
    psi = np.array(result["final_state_real"]) + 1j * np.array(
        result["final_state_imag"]
    )
    assert result["energy"] == pytest.approx(
        np.vdot(psi, qml.matrix(hamiltonian) @ psi).real
    )
    for params, energy in zip(result["params_history"], result["energies"]):
        assert energy == pytest.approx(np.cos(params[0]))


def test_adapt_inner_energy_matches_updated_parameters():
    theta, energies = _inner_optimize(
        energy_qnode=lambda x: qml.numpy.sum(x**2),
        theta_init=[1.0],
        optimizer_name="GradientDescent",
        stepsize=0.1,
        steps=1,
    )
    assert energies == pytest.approx([1.0, 0.64])
    assert energies[-1] == pytest.approx(float(qml.numpy.sum(theta**2)))


def test_old_cache_signatures_are_invalidated():
    config = {"molecule": "H2", "steps": 2}
    assert stable_hash_cfg(config) != stable_hash_dict(config)


@pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
def test_mapped_registry_summary_counts_electrons(mapping):
    row = summarize_registry_coverage(systems=["H2"], mapping=mapping)[0]
    assert row["num_electrons"] == 2


@pytest.mark.parametrize("mapping", ["parity", "bravyi_kitaev"])
def test_encoding_non_power_of_two_register(mapping):
    for n in [3, 6]:
        encoding = qml.matrix(apply_encoding, wire_order=range(n))(range(n), mapping)
        for electrons in range(1, n + 1):
            occupied = qml.qchem.hf_state(electrons, n)
            expected = qml.qchem.hf_state(electrons, n, basis=mapping)
            index = int("".join(map(str, occupied)), 2)
            assert np.argmax(encoding[:, index]) == int("".join(map(str, expected)), 2)


@pytest.mark.parametrize(
    "runner,options,key",
    [
        (run_adapt_vqe, {"max_ops": 1, "inner_steps": 1}, "energy"),
        (run_ssvqe, {"num_states": 2, "steps": 1}, "final_energies_sorted"),
        (run_vqd, {"num_states": 2, "steps": 1}, "energies_per_state"),
    ],
)
@pytest.mark.filterwarnings("error::numpy.exceptions.ComplexWarning")
def test_other_ucc_workflows_preserve_mapping_energies(runner, options, key):
    cfg = dict(
        molecule="H2",
        optimizer_name="GradientDescent",
        plot=False,
        force=True,
        **options,
    )
    original = runner(**cfg)
    mapped = runner(**cfg, mapping="parity")
    np.testing.assert_allclose(mapped[key], original[key], atol=1e-10)
