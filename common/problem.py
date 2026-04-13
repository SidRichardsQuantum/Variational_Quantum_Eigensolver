"""
common.problem
==============

Shared problem-resolution helpers for chemistry and expert-mode workflows.
"""

# ruff: noqa: I001

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from . import mpl_env as _mpl_env  # noqa: F401

import pennylane as qml

from .hamiltonian import build_hamiltonian, resolve_active_space


@dataclass(frozen=True)
class ResolvedProblem:
    hamiltonian: qml.Hamiltonian
    num_qubits: int
    reference_state: np.ndarray | None
    molecule_label: str
    symbols: list[str]
    coordinates: np.ndarray
    basis: str
    charge: int
    mapping: str
    unit: str
    active_electrons: int | None
    active_orbitals: int | None
    cacheable: bool


def resolve_problem(
    *,
    molecule: str = "H2",
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    charge: int = 0,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    hamiltonian: qml.Hamiltonian | None = None,
    num_qubits: int | None = None,
    reference_state=None,
    default_reference_state: bool = False,
    require_reference_state: bool = False,
    reference_name: str = "reference_state",
) -> ResolvedProblem:
    mapping_norm = str(mapping).strip().lower()
    basis_norm = str(basis).strip().lower()
    unit_norm = str(unit).strip().lower()
    charge_int = int(charge)

    if hamiltonian is not None:
        H = hamiltonian
        wire_order = list(H.wires)
        inferred_qubits = len(wire_order)
        resolved_num_qubits = (
            int(num_qubits) if num_qubits is not None else inferred_qubits
        )
        if resolved_num_qubits < inferred_qubits:
            raise ValueError(
                "num_qubits cannot be smaller than the number of wires used by the "
                "provided Hamiltonian."
            )
        if wire_order != list(range(inferred_qubits)):
            H = H.map_wires({w: i for i, w in enumerate(wire_order)})

        if reference_state is None:
            if require_reference_state:
                raise ValueError(
                    f"{reference_name} is required when hamiltonian is provided."
                )
            resolved_reference = (
                np.zeros(resolved_num_qubits, dtype=int)
                if default_reference_state
                else None
            )
        else:
            resolved_reference = np.array(reference_state, dtype=int)

        if (
            resolved_reference is not None
            and len(resolved_reference) != resolved_num_qubits
        ):
            raise ValueError(
                f"{reference_name} length must match num_qubits in Hamiltonian mode."
            )

        return ResolvedProblem(
            hamiltonian=H,
            num_qubits=resolved_num_qubits,
            reference_state=resolved_reference,
            molecule_label=str(molecule).strip() or "hamiltonian",
            symbols=list(symbols) if symbols is not None else [],
            coordinates=(
                np.array(coordinates, dtype=float)
                if coordinates is not None
                else np.array([], dtype=float)
            ),
            basis=basis_norm,
            charge=charge_int,
            mapping=mapping_norm,
            unit=unit_norm,
            active_electrons=(
                None if active_electrons is None else int(active_electrons)
            ),
            active_orbitals=(None if active_orbitals is None else int(active_orbitals)),
            cacheable=False,
        )

    if num_qubits is not None:
        raise ValueError("num_qubits is only supported when hamiltonian is provided.")
    if reference_state is not None:
        raise ValueError(
            f"{reference_name} is only supported when hamiltonian is provided."
        )

    if symbols is not None and coordinates is not None:
        (
            H,
            resolved_num_qubits,
            hf_state,
            symbols_out,
            coordinates_out,
            basis_out,
            charge_out,
            unit_out,
        ) = build_hamiltonian(
            molecule=None,
            symbols=list(symbols),
            coordinates=np.array(coordinates, dtype=float),
            charge=charge_int,
            basis=basis_norm,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            mapping=mapping_norm,
            unit=unit_norm,
            return_metadata=True,
        )
        molecule_label = str(molecule).strip() or "molecule"
    else:
        (
            H,
            resolved_num_qubits,
            hf_state,
            symbols_out,
            coordinates_out,
            basis_out,
            charge_out,
            unit_out,
        ) = build_hamiltonian(
            molecule=str(molecule),
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
            mapping=mapping_norm,
            unit=unit_norm,
            return_metadata=True,
        )
        molecule_label = str(molecule).strip()

    _, _, (resolved_active_electrons, resolved_active_orbitals) = resolve_active_space(
        symbols=list(symbols_out),
        coordinates=np.array(coordinates_out, dtype=float),
        charge=int(charge_out),
        basis=str(basis_out).strip().lower(),
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    return ResolvedProblem(
        hamiltonian=H,
        num_qubits=int(resolved_num_qubits),
        reference_state=np.array(hf_state, dtype=int),
        molecule_label=molecule_label,
        symbols=list(symbols_out),
        coordinates=np.array(coordinates_out, dtype=float),
        basis=str(basis_out).strip().lower(),
        charge=int(charge_out),
        mapping=mapping_norm,
        unit=str(unit_out).strip().lower(),
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        cacheable=True,
    )
