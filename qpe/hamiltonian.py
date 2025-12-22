"""
qpe.hamiltonian
---------------
QPE-facing Hamiltonian utilities.

This module is a *thin compatibility layer* over vqe_qpe_common.

Public API:
    - build_hamiltonian(molecule: str) -> (H, n_qubits, hf_state, symbols, coordinates, basis, charge)

Rationale:
    QPE needs:
      (1) the molecular Hamiltonian (qubit-mapped)
      (2) the Hartree–Fock reference state (bitstring)
    Both are produced by vqe_qpe_common.hamiltonian.build_hamiltonian.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pennylane as qml

from vqe_qpe_common.geometry import generate_geometry as _common_generate_geometry
from vqe_qpe_common.hamiltonian import build_hamiltonian as _common_build_hamiltonian
from vqe_qpe_common.molecules import MOLECULES, get_molecule_config


def _normalise_static_key(molecule: str) -> str:
    """
    Normalise molecule name for registry lookups.

    Accepts aliases:
      - "H3PLUS", "H3_PLUS" -> "H3+"
    """
    key = str(molecule).strip()
    up = key.upper().replace(" ", "")

    if up in {"H3PLUS", "H3_PLUS"}:
        return "H3+"

    if key in MOLECULES:
        return key

    # Case-insensitive match fallback
    for k in MOLECULES.keys():
        if k.upper() == up:
            return k

    raise ValueError(
        f"Unsupported molecule '{molecule}'. Available static presets: {list(MOLECULES.keys())} "
        "or parametric: H2_BOND, H3+_BOND, LiH_BOND, H2O_ANGLE."
    )


def build_hamiltonian(
    molecule: str,
) -> Tuple[qml.Hamiltonian, int, np.ndarray, List[str], np.ndarray, str, int]:
    """
    Build the molecular Hamiltonian + HF state for QPE.

    Parameters
    ----------
    molecule:
        Molecule identifier, e.g. "H2", "LiH", "H2O", "H3+",
        or parametric tags "H2_BOND", "LiH_BOND", "H2O_ANGLE", "H3+_BOND".

    Returns
    -------
    (H, n_qubits, hf_state, symbols, coordinates, basis, charge)
    """
    mol = str(molecule).strip()
    up = mol.upper()

    # Parametric molecules
    if "BOND" in up or "ANGLE" in up:
        # Default only used if caller provides just the tag.
        if up == "H2O_ANGLE":
            default_param = 104.5
        elif up in {"H3+_BOND", "H3PLUS_BOND", "H3_PLUS_BOND"}:
            default_param = 0.9
        else:
            default_param = 0.74

        symbols, coordinates = _common_generate_geometry(mol, float(default_param))

        # Charge: infer for the parametric H3+ variant; otherwise neutral.
        charge = (
            +1
            if up.startswith("H3+")
            or up.startswith("H3PLUS")
            or up.startswith("H3_PLUS")
            else 0
        )
        basis = "STO-3G"

    # Static molecules from registry
    else:
        key = _normalise_static_key(mol)
        cfg = get_molecule_config(key)
        symbols = list(cfg["symbols"])
        coordinates = np.array(cfg["coordinates"], dtype=float)
        charge = int(cfg["charge"])
        basis = str(cfg["basis"])

    H, n_qubits, hf_state = _common_build_hamiltonian(
        symbols=symbols,
        coordinates=coordinates,
        charge=charge,
        basis=basis,
    )

    return (
        H,
        int(n_qubits),
        np.array(hf_state, dtype=int),
        symbols,
        coordinates,
        basis,
        charge,
    )
