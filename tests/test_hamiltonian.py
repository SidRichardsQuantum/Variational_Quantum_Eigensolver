from __future__ import annotations

import pytest

from common.hamiltonian import build_hamiltonian, summarize_registry_coverage
from common.molecules import get_molecule_config
from common.units import ANGSTROM_PER_BOHR


def test_build_hamiltonian_h2() -> None:
    cfg = get_molecule_config("H2")
    hamiltonian, n_qubits, hf_state = build_hamiltonian(**cfg)

    assert n_qubits > 0
    assert len(hf_state) == n_qubits

    coeffs, ops = hamiltonian.terms()
    assert len(coeffs) == len(ops)
    assert len(coeffs) > 0
    assert all(hasattr(op, "wires") for op in ops)


def test_build_hamiltonian_selected_registry_molecules() -> None:
    for name in [
        "H",
        "H-",
        "He",
        "He+",
        "H2",
        "H2+",
        "H2-",
        "H3",
        "H3+",
        "Li",
        "Li+",
        "B",
        "B+",
        "C",
        "C+",
        "N",
        "N+",
        "O",
        "O+",
        "F",
        "F+",
        "Ne",
        "H4",
        "H4+",
        "Be",
        "Be+",
        "He2",
        "H5+",
        "H6",
        "LiH",
        "BeH2",
        "HeH+",
    ]:
        cfg = get_molecule_config(name)
        hamiltonian, n_qubits, hf_state = build_hamiltonian(**cfg)

        assert n_qubits > 0
        assert len(hf_state) == n_qubits
        assert len(hamiltonian) > 0


def test_registry_mode_rejects_basis_and_charge_overrides() -> None:
    with pytest.raises(ValueError, match="Registry mode does not accept `basis`"):
        build_hamiltonian(molecule="H2", basis="6-31g")

    with pytest.raises(ValueError, match="Registry mode does not accept `charge`"):
        build_hamiltonian(molecule="H2", charge=1)


def test_registry_mode_uses_open_shell_multiplicity_defaults() -> None:
    hamiltonian, n_qubits, hf_state = build_hamiltonian(molecule="H2+")

    assert n_qubits == 4
    assert len(hf_state) == n_qubits
    assert int(sum(hf_state)) == 1
    assert len(hamiltonian) > 0


def test_registry_mode_converts_coordinates_for_bohr_metadata() -> None:
    (
        _hamiltonian,
        _n_qubits,
        _hf_state,
        _symbols,
        coordinates,
        _basis,
        _charge,
        unit_out,
    ) = build_hamiltonian(molecule="H2", unit="bohr", return_metadata=True)

    bond_length = float(coordinates[1, 2] - coordinates[0, 2])
    assert unit_out == "bohr"
    assert bond_length == pytest.approx(0.7414 / ANGSTROM_PER_BOHR)


def test_explicit_mode_preserves_input_coordinate_unit_metadata() -> None:
    (
        _hamiltonian,
        _n_qubits,
        _hf_state,
        _symbols,
        coordinates,
        _basis,
        _charge,
        unit_out,
    ) = build_hamiltonian(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        charge=0,
        basis="sto-3g",
        unit="bohr",
        return_metadata=True,
    )

    assert unit_out == "bohr"
    assert coordinates[1, 2] == pytest.approx(1.4)


def test_invalid_coordinate_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported coordinate unit"):
        build_hamiltonian(molecule="H2", unit="nanometers")


def test_explicit_mode_accepts_multiplicity() -> None:
    hamiltonian, n_qubits, hf_state = build_hamiltonian(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]],
        charge=1,
        multiplicity=2,
        basis="sto-3g",
    )

    assert n_qubits == 4
    assert int(sum(hf_state)) == 1
    assert len(hamiltonian) > 0


def test_open_shell_registry_build_avoids_fallback_warning(capsys) -> None:
    hamiltonian, n_qubits, hf_state = build_hamiltonian(molecule="H2+")

    captured = capsys.readouterr()
    assert "retrying with OpenFermion" not in captured.out
    assert n_qubits == 4
    assert int(sum(hf_state)) == 1
    assert len(hamiltonian) > 0


def test_summarize_registry_coverage_returns_expected_rows() -> None:
    rows = summarize_registry_coverage(systems=["H2", "H2+"])

    assert [row["molecule"] for row in rows] == ["H2", "H2+"]

    h2_row, h2_plus_row = rows
    assert h2_row["num_qubits"] == 4
    assert h2_row["num_electrons"] == 2
    assert h2_plus_row["num_qubits"] == 4
    assert h2_plus_row["num_electrons"] == 1
    assert "exact_ground_energy" in h2_row


def test_active_space_reduces_lih_qubit_count() -> None:
    full_hamiltonian, full_qubits, full_hf = build_hamiltonian(molecule="LiH")
    active_hamiltonian, active_qubits, active_hf = build_hamiltonian(
        molecule="LiH",
        active_electrons=2,
        active_orbitals=2,
    )

    assert full_qubits > active_qubits
    assert active_qubits == 4
    assert len(active_hf) == active_qubits
    assert len(active_hamiltonian) > 0
    assert len(full_hamiltonian) > 0
