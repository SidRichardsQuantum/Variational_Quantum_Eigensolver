from __future__ import annotations

import pytest

from common.hamiltonian import build_hamiltonian
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
    for name in ["H2", "H3+", "LiH", "BeH2", "HeH+"]:
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
