"""
vqe.hamiltonian
---------------
VQE-facing Hamiltonian and geometry utilities.

This module is a thin compatibility layer over vqe_qpe_common. It preserves the
historical VQE API:

    - MOLECULES
    - generate_geometry(...)
    - build_hamiltonian(molecule, mapping="jordan_wigner")

returning:
    (H, num_qubits, symbols, coordinates, basis)

Single source of truth:
    - molecule registry:    vqe_qpe_common.molecules
    - geometry generators:  vqe_qpe_common.geometry
    - Hamiltonian builder:  vqe_qpe_common.hamiltonian.build_hamiltonian
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pennylane as qml

from vqe_qpe_common.geometry import generate_geometry as _common_generate_geometry
from vqe_qpe_common.hamiltonian import build_hamiltonian as _build_common_hamiltonian
from vqe_qpe_common.molecules import MOLECULES as _COMMON_MOLECULES
from vqe_qpe_common.molecules import get_molecule_config

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

    Delegates to vqe_qpe_common.geometry.generate_geometry (single source of truth).
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


def _apply_mapping_if_possible(
    H: qml.Hamiltonian,
    mapping: str,
) -> qml.Hamiltonian:
    """
    Attempt to convert a qubit Hamiltonian to a different fermion-to-qubit mapping.

    Notes
    -----
    PennyLane's qchem.molecular_hamiltonian can apply mapping directly in some versions,
    but vqe_qpe_common.hamiltonian intentionally does not (to remain minimal and QPE-friendly).

    Here we attempt a best-effort conversion using OpenFermion intermediates.
    If conversion is unavailable, we return H unchanged and emit a warning.
    """
    mapping = str(mapping).lower().strip()
    if mapping in {"jordan_wigner", "jw"}:
        return H

    # Best-effort OpenFermion conversion path
    try:
        import openfermion  # noqa: F401

        try:
            from pennylane.qchem.convert import to_openfermion
        except Exception:
            to_openfermion = None

        if to_openfermion is None:
            raise RuntimeError(
                "PennyLane 'to_openfermion' not available in this environment."
            )

        raise RuntimeError(
            "Exact remapping requires constructing the Hamiltonian with mapping at build time. "
            "Use vqe.hamiltonian.build_hamiltonian(..., mapping=...) with a PennyLane version "
            "that supports qchem.molecular_hamiltonian(mapping=...)."
        )

    except Exception as exc:
        print(
            f"⚠️  Mapping '{mapping}' requested but could not be applied via conversion path.\n"
            f"    Proceeding with the default mapping used by the Hamiltonian builder.\n"
            f"    Details: {exc}"
        )
        return H


def build_hamiltonian(molecule: str, mapping: str = "jordan_wigner"):
    """
    Construct the qubit Hamiltonian for a given molecule.

    Supports:
        - Static presets in vqe_qpe_common.molecules.MOLECULES
        - Parametric variants handled by vqe_qpe_common.geometry.generate_geometry

    Parameters
    ----------
    molecule:
        Molecule identifier, e.g. "H2", "LiH", "H2O", "H3+",
        or parametric "H2_BOND", "LiH_BOND", "H2O_ANGLE", "H3+_BOND".
    mapping:
        Fermion-to-qubit mapping scheme label (best-effort; see notes).

    Returns
    -------
    (H, num_qubits, symbols, coordinates, basis)
    """
    mol = str(molecule).strip()
    up = mol.upper()

    # Parametric tags: if caller passes only the tag, choose a default parameter
    if "BOND" in up or "ANGLE" in up:
        if up == "H2O_ANGLE":
            default_param = 104.5
        elif up in {"H3+_BOND", "H3PLUS_BOND", "H3_PLUS_BOND"}:
            default_param = 0.9
        else:
            default_param = 0.74

        symbols, coordinates = generate_geometry(mol, default_param)
        charge = +1 if up.startswith(("H3+", "H3PLUS", "H3_PLUS")) else 0
        basis = "STO-3G"
    else:
        key = _normalise_static_key(mol)
        cfg = get_molecule_config(key)
        symbols = list(cfg["symbols"])
        coordinates = np.array(cfg["coordinates"], dtype=float)
        charge = int(cfg["charge"])
        basis = str(cfg["basis"])

    # Common build: returns (H, n_qubits, hf_state)
    H, num_qubits, _hf_state = _build_common_hamiltonian(
        symbols=symbols,
        coordinates=np.array(coordinates, dtype=float),
        charge=charge,
        basis=basis,
    )

    # Best-effort mapping application
    H_mapped = _apply_mapping_if_possible(H, mapping=mapping)

    return (
        H_mapped,
        int(num_qubits),
        list(symbols),
        np.array(coordinates, dtype=float),
        str(basis),
    )


def hartree_fock_state(
    molecule: str,
    *,
    mapping: str = "jordan_wigner",
) -> np.ndarray:
    """
    Return the Hartree–Fock occupation bitstring for the molecule.

    This keeps vqe.hamiltonian.build_hamiltonian(...) backwards compatible
    (it still returns (H, n_qubits, symbols, coordinates, basis)) while enabling
    chemistry-aware routines (SSVQE/VQD helpers) to obtain HF when needed.
    """
    mol = str(molecule).strip()
    up = mol.upper()

    if "BOND" in up or "ANGLE" in up:
        if up == "H2O_ANGLE":
            default_param = 104.5
        elif up in {"H3+_BOND", "H3PLUS_BOND", "H3_PLUS_BOND"}:
            default_param = 0.9
        else:
            default_param = 0.74

        symbols, coordinates = generate_geometry(mol, default_param)
        charge = +1 if up.startswith(("H3+", "H3PLUS", "H3_PLUS")) else 0
        basis = "STO-3G"
    else:
        key = _normalise_static_key(mol)
        cfg = get_molecule_config(key)
        symbols = list(cfg["symbols"])
        coordinates = np.array(cfg["coordinates"], dtype=float)
        charge = int(cfg["charge"])
        basis = str(cfg["basis"])

    # We call the common builder to get consistent n_qubits and HF state.
    # (Mapping is best-effort in your layer; HF is defined with respect to n_qubits.)
    _H, n_qubits, hf = _build_common_hamiltonian(
        symbols=symbols,
        coordinates=np.array(coordinates, dtype=float),
        charge=charge,
        basis=basis,
        mapping=None,  # HF state depends on n_qubits; common builder handles it.
    )

    return np.array(hf, dtype=int)
