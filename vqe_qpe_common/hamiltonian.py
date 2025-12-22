"""
vqe_qpe_common.hamiltonian
==========================

Shared Hamiltonian construction used by VQE, QPE, and future solvers.

Design goals
------------
1) Single source of truth for molecular Hamiltonian construction.
2) Optional support for fermion-to-qubit mappings (JW/BK/Parity) when available.
3) OpenFermion fallback when the default backend fails.
4) Hartree–Fock state utilities separated from Hamiltonian construction.

Backwards compatibility
-----------------------
We keep a legacy-style:
    build_hamiltonian(symbols, coordinates, charge, basis, mapping=None)
that returns (H, n_qubits, hf_state)

New recommended API:
    build_molecular_hamiltonian(...)
    hartree_fock_state_from_molecule(...)
    build_from_molecule_name(...)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import qchem

from vqe_qpe_common.molecules import get_molecule_config


# ---------------------------------------------------------------------
# Hartree–Fock state helpers
# ---------------------------------------------------------------------
def hartree_fock_state_from_molecule(
    *,
    symbols: list[str],
    coordinates: np.ndarray,
    charge: int,
    basis: str,
    n_qubits: int,
) -> np.ndarray:
    """
    Compute Hartree–Fock occupation bitstring using PennyLane-qchem Molecule.

    This avoids hand-rolled atomic-number tables and is robust across
    the supported element set.

    Returns
    -------
    np.ndarray
        0/1 HF bitstring of length n_qubits.
    """
    # Ensure plain array for qchem
    coords = np.array(coordinates, dtype=float)

    try:
        mol = qchem.Molecule(symbols, coords, charge=charge, basis=basis)
    except TypeError:
        mol = qchem.Molecule(symbols, coords, charge=charge)

    electrons = int(mol.n_electrons)
    return qchem.hf_state(electrons, n_qubits)


# ---------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------
def build_molecular_hamiltonian(
    *,
    symbols: list[str],
    coordinates: np.ndarray,
    charge: int,
    basis: str,
    mapping: Optional[str] = None,
    unit: str = "angstrom",
    method_fallback: bool = True,
) -> Tuple[qml.Hamiltonian, int]:
    """
    Build a molecular qubit Hamiltonian using PennyLane-qchem.

    Parameters
    ----------
    symbols, coordinates, charge, basis:
        Standard molecular inputs.
    mapping:
        Optional fermion-to-qubit mapping ("jordan_wigner", "bravyi_kitaev", "parity").
        If the installed PennyLane version does not support mapping=, we fall back
        gracefully to the default (typically Jordan–Wigner).
    unit:
        Passed through to qchem.molecular_hamiltonian.
    method_fallback:
        If True, retry with method="openfermion" if primary backend fails.

    Returns
    -------
    (H, n_qubits)
    """
    coords = np.array(coordinates, dtype=float)
    mapping_kw = None if mapping is None else str(mapping).strip().lower()

    # --- Attempt 1: default qchem backend, with mapping if supported ---
    try:
        kwargs: Dict[str, Any] = dict(
            symbols=symbols,
            coordinates=coords,
            charge=int(charge),
            basis=basis,
            unit=unit,
        )
        if mapping_kw is not None:
            kwargs["mapping"] = mapping_kw

        H, n_qubits = qchem.molecular_hamiltonian(**kwargs)
        return H, int(n_qubits)

    except TypeError as exc_type:
        # Retry without mapping if that was provided.
        if mapping_kw is not None:
            try:
                kwargs = dict(
                    symbols=symbols,
                    coordinates=coords,
                    charge=int(charge),
                    basis=basis,
                    unit=unit,
                )
                H, n_qubits = qchem.molecular_hamiltonian(**kwargs)
                return H, int(n_qubits)
            except Exception:
                # Fall through to global fallback below
                e_primary: Exception = exc_type
        else:
            e_primary = exc_type

    except Exception as exc_primary:
        e_primary = exc_primary

    # --- Attempt 2: optional OpenFermion fallback ---
    if not method_fallback:
        raise RuntimeError(
            "Failed to construct Hamiltonian (fallback disabled).\n"
            f"Primary error: {e_primary}"
        )

    print("⚠️ Default PennyLane-qchem backend failed — retrying with OpenFermion...")
    try:
        kwargs = dict(
            symbols=symbols,
            coordinates=coords,
            charge=int(charge),
            basis=basis,
            unit=unit,
            method="openfermion",
        )
        if mapping_kw is not None:
            try:
                kwargs["mapping"] = mapping_kw
                H, n_qubits = qchem.molecular_hamiltonian(**kwargs)
                return H, int(n_qubits)
            except TypeError:
                kwargs.pop("mapping", None)

        H, n_qubits = qchem.molecular_hamiltonian(**kwargs)
        return H, int(n_qubits)

    except Exception as e_fallback:
        raise RuntimeError(
            "Failed to construct Hamiltonian.\n"
            f"Primary error: {e_primary}\n"
            f"Fallback error: {e_fallback}"
        )


def build_from_molecule_name(
    name: str,
    *,
    mapping: Optional[str] = None,
    unit: str = "angstrom",
) -> Tuple[qml.Hamiltonian, int, Dict[str, Any]]:
    """
    Convenience wrapper for the common molecule registry.

    Returns
    -------
    (H, n_qubits, cfg)
        cfg is the molecule config dict from vqe_qpe_common.molecules.
    """
    cfg = get_molecule_config(name)
    H, n_qubits = build_molecular_hamiltonian(
        symbols=cfg["symbols"],
        coordinates=cfg["coordinates"],
        charge=cfg["charge"],
        basis=cfg["basis"],
        mapping=mapping,
        unit=unit,
    )
    return H, n_qubits, cfg


# ---------------------------------------------------------------------
# Backwards-compatible wrapper (legacy signature)
# ---------------------------------------------------------------------
def build_hamiltonian(
    symbols,
    coordinates,
    charge,
    basis,
    mapping: Optional[str] = None,
) -> Tuple[qml.Hamiltonian, int, np.ndarray]:
    """
    Legacy-style builder used by existing QPE code:

        build_hamiltonian(symbols, coordinates, charge, basis) -> (H, n_qubits, hf_state)

    Now implemented on top of the new shared APIs.
    """
    H, n_qubits = build_molecular_hamiltonian(
        symbols=list(symbols),
        coordinates=np.array(coordinates, dtype=float),
        charge=int(charge),
        basis=str(basis),
        mapping=mapping,
    )
    hf_state = hartree_fock_state_from_molecule(
        symbols=list(symbols),
        coordinates=np.array(coordinates, dtype=float),
        charge=int(charge),
        basis=str(basis),
        n_qubits=int(n_qubits),
    )
    return H, int(n_qubits), hf_state
