"""
vqe.hamiltonian
---------------
VQE-facing Hamiltonian and geometry utilities.

This module is a thin compatibility layer over common. It preserves the
historical VQE API:

    - MOLECULES
    - generate_geometry(...)
    - build_hamiltonian(molecule, mapping="jordan_wigner")

returning:
    (H, qubits, hf_state, symbols, coordinates, basis, charge, unit)

Single source of truth:
    - molecule registry:    common.molecules
    - geometry generators:  common.geometry
    - Hamiltonian builder:  common.hamiltonian.build_hamiltonian
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pennylane as qml

from common.geometry import generate_geometry as _common_generate_geometry
from common.hamiltonian import build_hamiltonian as _build_common_hamiltonian
from common.molecules import MOLECULES as _COMMON_MOLECULES
from common.molecules import get_molecule_config

# ---------------------------------------------------------------------
# Public re-export: molecule registry (backwards compatible)
# ---------------------------------------------------------------------
MOLECULES = _COMMON_MOLECULES


# ---------------------------------------------------------------------
# Compatibility: parametric geometry generation
# ---------------------------------------------------------------------
def generate_geometry(
    molecule: str, param_value: float
) -> Tuple[list[str], np.ndarray]:
    """
    Compatibility wrapper.

    Delegates to common.geometry.generate_geometry (single source of truth).
    """
    name = str(molecule).strip()
    return _common_generate_geometry(name, float(param_value))


def _normalise_static_key(molecule: str) -> str:
    """
    Normalise molecule name for static registry lookups.

    - Accepts "H3PLUS", "H3_PLUS" as aliases for "H3+"
    - Case-insensitive lookup fallback
    """
    key = str(molecule).strip()
    up = key.upper().replace(" ", "")

    if up in {"H3PLUS", "H3_PLUS"}:
        return "H3+"

    if key in MOLECULES:
        return key

    for k in MOLECULES.keys():
        if k.upper().replace(" ", "") == up:
            return k

    raise ValueError(
        f"Unsupported molecule '{molecule}'. "
        f"Available static presets: {list(MOLECULES.keys())}, "
        "or parametric: H2_BOND, H3+_BOND, LiH_BOND, H2O_ANGLE."
    )


def build_hamiltonian(
    molecule: str,
    mapping: Optional[str] = "jordan_wigner",
    unit: str = "angstrom",
) -> Tuple[qml.Hamiltonian, int, np.ndarray, List[str], np.ndarray, str, int, str]:
    """
    Construct the qubit Hamiltonian for a given molecule.

    Returns
    -------
    (H, num_qubits, hf_state, symbols, coordinates, basis, charge, unit_out)
    """
    mol = str(molecule).strip()
    up = mol.upper().replace(" ", "")

    # Parametric tags
    if "BOND" in up or "ANGLE" in up:
        if up == "H2O_ANGLE":
            default_param = 104.5
        elif up in {"H3+_BOND", "H3PLUS_BOND", "H3_PLUS_BOND"}:
            default_param = 0.9
        else:
            default_param = 0.74

        symbols, coordinates = generate_geometry(mol, float(default_param))
        charge = +1 if up.startswith(("H3+", "H3PLUS", "H3_PLUS")) else 0
        basis = "sto-3g"
    else:
        key = _normalise_static_key(mol)
        cfg = get_molecule_config(key)
        symbols = list(cfg["symbols"])
        coordinates = np.array(cfg["coordinates"], dtype=float)
        charge = int(cfg["charge"])
        basis = str(cfg["basis"]).strip().lower()

    mapping_norm = None if mapping is None else str(mapping).strip().lower()
    unit_norm = str(unit).strip().lower()

    H, qubits, hf_state, sym_out, coords_out, basis_out, charge_out, unit_out = (
        _build_common_hamiltonian(
            symbols=list(symbols),
            coordinates=np.array(coordinates, dtype=float),
            charge=int(charge),
            basis=str(basis),
            mapping=str(mapping_norm) if mapping_norm is not None else "jordan_wigner",
            unit=str(unit_norm),
            return_metadata=True,
        )
    )

    return (
        H,
        int(qubits),
        np.array(hf_state, dtype=int),
        list(sym_out),
        np.array(coords_out, dtype=float),
        str(basis_out),
        int(charge_out),
        str(unit_out),
    )


def hartree_fock_state(
    molecule: str,
    *,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
) -> np.ndarray:
    """
    Return the Hartree–Fock occupation bitstring for the molecule.
    """
    H, qubits, hf_state, symbols, coordinates, basis, charge, unit_out = (
        build_hamiltonian(
            molecule=molecule,
            mapping=mapping,
            unit=unit,
        )
    )
    return np.array(hf_state, dtype=int)
