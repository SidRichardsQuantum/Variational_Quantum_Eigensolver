"""
common.hamiltonian
==========================

Shared Hamiltonian construction used by VQE, QPE, and future solvers.

Design goals
------------
1) Single source of truth for molecular Hamiltonian construction.
2) Optional support for fermion-to-qubit mappings (JW/BK/Parity) when available.
3) OpenFermion fallback when the default backend fails.
4) Hartree–Fock state utilities separated from Hamiltonian construction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pennylane as qml
from pennylane import qchem

from common.geometry import generate_geometry
from common.molecules import get_molecule_config


def _normalise_static_key(name: str) -> str:
    """
    Normalise common user spellings/aliases to canonical molecule keys in MOLECULES.

    Examples
    --------
    "h2" -> "H2"
    "H3PLUS" / "H3_PLUS" -> "H3+"
    """
    s = str(name).strip()
    if not s:
        raise ValueError("molecule name must be a non-empty string")

    up = s.upper().replace(" ", "").replace("-", "_")

    # Canonicalise H3+ aliases
    if up in {"H3+", "H3PLUS", "H3_PLUS"}:
        return "H3+"

    # Preserve + for other ions if user included it
    # but normalise plain molecule strings by stripping underscores
    s2 = s.replace("_", "").strip()

    # Common simple molecules
    if s2.upper() == "H2":
        return "H2"
    if s2.upper() == "LIH":
        return "LiH"
    if s2.upper() == "H2O":
        return "H2O"
    if s2.upper() == "HEH+" or up == "HEH+":
        return "HeH+"
    if s2.upper() == "BEH2":
        return "BeH2"
    if s2.upper() == "H4":
        return "H4"

    # Fall back to original (registry will raise if unknown)
    return s


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
        cfg is the molecule config dict from common.molecules.
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
# Hamiltonian + HF state (single public entrypoint)
# ---------------------------------------------------------------------
def build_hamiltonian(
    molecule: Optional[str] = None,
    coordinates: Optional[np.ndarray] = None,
    symbols: Optional[list[str]] = None,
    *,
    charge: Optional[int] = None,
    basis: Optional[str] = None,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    return_metadata: bool = False,
):
    """
    Unified Hamiltonian entrypoint.

    Supported call styles
    ---------------------
    1) Registry / tag mode:
        build_hamiltonian("H2", mapping="jordan_wigner", unit="angstrom")

    2) Explicit molecule mode (used by tests and geometry scans):
        build_hamiltonian(symbols=[...], coordinates=array(...), charge=0, basis="sto-3g")

    Notes
    -----
    - This function intentionally does NOT treat non-string `molecule` as a registry key.
      Tests often pass atoms/coords positionally; that is handled here by interpreting
      (molecule, coordinates) as (symbols, coordinates) when `molecule` is a sequence.

    Returns
    -------
    Default (return_metadata=False):
        (H, n_qubits, hf_state)

    With return_metadata=True:
        (H, n_qubits, hf_state, symbols, coordinates, basis, charge, unit_out)
    """
    unit_norm = str(unit).strip().lower()
    mapping_norm = str(mapping).strip().lower()

    # ------------------------------------------------------------------
    # Back-compat positional explicit mode:
    #   build_hamiltonian(symbols, coordinates, charge=..., basis=...)
    # where `symbols` may have been passed as the first positional arg.
    # ------------------------------------------------------------------
    if symbols is None and coordinates is not None and molecule is not None:
        # If molecule is not a string, interpret it as the symbols list.
        if not isinstance(molecule, str):
            symbols = molecule  # type: ignore[assignment]
            molecule = None

    # ------------------------------------------------------------
    # Explicit molecule mode (symbols + coordinates provided)
    # ------------------------------------------------------------
    if symbols is not None and coordinates is not None:
        if charge is None:
            raise TypeError("build_hamiltonian(...): missing required keyword 'charge'")
        if basis is None:
            raise TypeError("build_hamiltonian(...): missing required keyword 'basis'")

        sym = list(symbols)
        coords = np.array(coordinates, dtype=float)
        chg = int(charge)
        bas = str(basis).strip().lower()

        H, n_qubits = build_molecular_hamiltonian(
            symbols=sym,
            coordinates=coords,
            charge=chg,
            basis=bas,
            mapping=mapping_norm,
            unit=unit_norm,
            method_fallback=True,
        )
        hf_state = hartree_fock_state_from_molecule(
            symbols=sym,
            coordinates=coords,
            charge=chg,
            basis=bas,
            n_qubits=int(n_qubits),
        )

        hf_state = np.array(hf_state, dtype=int)

        if not return_metadata:
            return H, int(n_qubits), hf_state

        return (
            H,
            int(n_qubits),
            hf_state,
            sym,
            np.array(coords, dtype=float),
            bas,
            chg,
            unit_norm,
        )

    # ------------------------------------------------------------
    # Registry / tag mode (molecule name)
    # ------------------------------------------------------------
    if molecule is None:
        raise TypeError(
            "build_hamiltonian(...): provide either a molecule name string via `molecule`, "
            "or both `symbols` and `coordinates` (plus `charge` and `basis`)."
        )

    if not isinstance(molecule, str):
        raise TypeError(
            "build_hamiltonian(...): `molecule` must be a string in registry/tag mode. "
            "If you intended explicit mode, pass `symbols=` and `coordinates=` (and `charge`, `basis`)."
        )

    mol = molecule.strip()
    if not mol:
        raise ValueError("molecule must be a non-empty string")

    up = mol.upper()

    # Parametric tags: choose a default parameter
    if ("BOND" in up) or ("ANGLE" in up):
        if up == "H2O_ANGLE":
            default_param = 104.5  # degrees
        elif up in {"H3+_BOND", "H3PLUS_BOND", "H3_PLUS_BOND"}:
            default_param = 0.9
        else:
            default_param = 0.74

        sym, coords = generate_geometry(mol, float(default_param))
        chg = +1 if up.startswith(("H3+", "H3PLUS", "H3_PLUS")) else 0
        bas = "sto-3g"
    else:
        key = _normalise_static_key(mol)
        cfg = get_molecule_config(key)
        sym = list(cfg["symbols"])
        coords = np.array(cfg["coordinates"], dtype=float)
        chg = int(cfg["charge"])
        bas = str(cfg["basis"]).strip().lower()

    H, n_qubits = build_molecular_hamiltonian(
        symbols=list(sym),
        coordinates=np.array(coords, dtype=float),
        charge=int(chg),
        basis=str(bas).lower(),
        mapping=mapping_norm,
        unit=unit_norm,
        method_fallback=True,
    )
    hf_state = hartree_fock_state_from_molecule(
        symbols=list(sym),
        coordinates=np.array(coords, dtype=float),
        charge=int(chg),
        basis=str(bas).lower(),
        n_qubits=int(n_qubits),
    )

    hf_state = np.array(hf_state, dtype=int)

    if not return_metadata:
        return H, int(n_qubits), hf_state

    return (
        H,
        int(n_qubits),
        hf_state,
        list(sym),
        np.array(coords, dtype=float),
        str(bas).lower(),
        int(chg),
        unit_norm,
    )
