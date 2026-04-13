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

# ruff: noqa: I001

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from common import mpl_env as _mpl_env  # noqa: F401

import pennylane as qml
from pennylane import qchem

from common.geometry import generate_geometry
from common.molecules import get_molecule_config
from common.units import convert_coordinates, convert_length, normalize_coordinate_unit


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

    # Canonicalise simple ionic aliases
    if up in {"H3+", "H3PLUS", "H3_PLUS"}:
        return "H3+"
    if up in {"H5+", "H5PLUS", "H5_PLUS"}:
        return "H5+"

    # Preserve + for other ions if user included it
    # but normalise plain molecule strings by stripping underscores
    s2 = s.replace("_", "").strip()

    # Common simple molecules
    if s2.upper() == "H2":
        return "H2"
    if s2.upper() == "H6":
        return "H6"
    if s2.upper() == "HE2":
        return "He2"
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
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
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

    electrons, spin_orbitals, _active_orbitals = resolve_active_space(
        symbols=symbols,
        coordinates=coords,
        charge=charge,
        basis=basis,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    return qchem.hf_state(electrons, spin_orbitals)


def _make_molecule(
    *,
    symbols: list[str],
    coordinates: np.ndarray,
    charge: int,
    basis: str,
):
    coords = np.array(coordinates, dtype=float)
    try:
        return qchem.Molecule(symbols, coords, charge=charge, basis=basis)
    except TypeError:
        return qchem.Molecule(symbols, coords, charge=charge)


def resolve_active_space(
    *,
    symbols: list[str],
    coordinates: np.ndarray,
    charge: int,
    basis: str,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
) -> tuple[int, int, tuple[int | None, int | None]]:
    """
    Resolve active-space settings into electron and spin-orbital counts.

    Returns
    -------
    (electrons, spin_orbitals, (resolved_active_electrons, resolved_active_orbitals))
    """
    mol = _make_molecule(
        symbols=list(symbols),
        coordinates=np.array(coordinates, dtype=float),
        charge=int(charge),
        basis=str(basis),
    )

    total_electrons = int(mol.n_electrons)
    total_orbitals = int(mol.n_orbitals)

    ae = None if active_electrons is None else int(active_electrons)
    ao = None if active_orbitals is None else int(active_orbitals)

    if ae is None and ao is None:
        return total_electrons, 2 * total_orbitals, (None, None)

    if ae is not None and ae <= 0:
        raise ValueError("active_electrons must be a positive integer when provided")
    if ao is not None and ao <= 0:
        raise ValueError("active_orbitals must be a positive integer when provided")

    core, active = qchem.active_space(
        total_electrons,
        total_orbitals,
        mult=1,
        active_electrons=ae,
        active_orbitals=ao,
    )
    resolved_active_orbitals = int(len(active))
    resolved_active_electrons = int(total_electrons - 2 * len(core))
    return (
        resolved_active_electrons,
        2 * resolved_active_orbitals,
        (resolved_active_electrons, resolved_active_orbitals),
    )


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
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
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
        Coordinate unit for `coordinates`. Supported values are `angstrom` and `bohr`.
        This affects geometry only. Energies returned by downstream solvers remain in Hartree.
    method_fallback:
        If True, retry with method="openfermion" if primary backend fails.

    Returns
    -------
    (H, n_qubits)
    """
    unit_norm = normalize_coordinate_unit(unit)
    coords = np.array(coordinates, dtype=float)
    mapping_kw = None if mapping is None else str(mapping).strip().lower()

    # --- Attempt 1: default qchem backend, with mapping if supported ---
    try:
        kwargs: Dict[str, Any] = dict(
            symbols=symbols,
            coordinates=coords,
            charge=int(charge),
            basis=basis,
            unit=unit_norm,
        )
        if active_electrons is not None:
            kwargs["active_electrons"] = int(active_electrons)
        if active_orbitals is not None:
            kwargs["active_orbitals"] = int(active_orbitals)
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
                    unit=unit_norm,
                )
                if active_electrons is not None:
                    kwargs["active_electrons"] = int(active_electrons)
                if active_orbitals is not None:
                    kwargs["active_orbitals"] = int(active_orbitals)
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
            unit=unit_norm,
            method="openfermion",
        )
        if active_electrons is not None:
            kwargs["active_electrons"] = int(active_electrons)
        if active_orbitals is not None:
            kwargs["active_orbitals"] = int(active_orbitals)
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
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
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
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
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
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
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
    - `unit` controls coordinate input/output units only (`angstrom` or `bohr`).
      Hamiltonian eigenvalues and solver energies remain in Hartree (Ha).
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
    unit_norm = normalize_coordinate_unit(unit)
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
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )
        hf_state = hartree_fock_state_from_molecule(
            symbols=sym,
            coordinates=coords,
            charge=chg,
            basis=bas,
            n_qubits=int(n_qubits),
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
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

    if basis is not None:
        raise ValueError(
            "Registry mode does not accept `basis`. Use explicit geometry mode "
            "with `symbols`, `coordinates`, `charge`, and `basis`."
        )
    if charge is not None:
        raise ValueError(
            "Registry mode does not accept `charge`. Use explicit geometry mode "
            "with `symbols`, `coordinates`, `charge`, and `basis`."
        )

    up = mol.upper()

    # Parametric tags: choose a default parameter
    if ("BOND" in up) or ("ANGLE" in up):
        if up == "H2O_ANGLE":
            default_param = 104.5  # degrees
        elif up in {"H3+_BOND", "H3PLUS_BOND", "H3_PLUS_BOND"}:
            default_param = 0.9
        else:
            default_param = 0.74

        if "BOND" in up:
            default_param = convert_length(
                float(default_param),
                from_unit="angstrom",
                to_unit=unit_norm,
            )

        sym, coords = generate_geometry(mol, float(default_param), unit=unit_norm)
        chg = +1 if up.startswith(("H3+", "H3PLUS", "H3_PLUS")) else 0
        bas = "sto-3g"
    else:
        key = _normalise_static_key(mol)
        cfg = get_molecule_config(key)
        sym = list(cfg["symbols"])
        coords = convert_coordinates(
            cfg["coordinates"],
            from_unit=str(cfg.get("unit", "angstrom")),
            to_unit=unit_norm,
        )
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
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    hf_state = hartree_fock_state_from_molecule(
        symbols=list(sym),
        coordinates=np.array(coords, dtype=float),
        charge=int(chg),
        basis=str(bas).lower(),
        n_qubits=int(n_qubits),
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
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


def get_exact_spectrum(
    molecule: str = "H2",
    *,
    k: int = 10,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
) -> list[float]:
    """
    Return the lowest-k exact eigenvalues (Ha) for the qubit Hamiltonian of `molecule`.

    This is a package-level helper so notebooks do not need to call build_hamiltonian/qml.matrix directly.
    """
    import numpy as _np
    import pennylane as _qml

    H, *_ = build_hamiltonian(
        str(molecule),
        mapping=str(mapping).strip().lower(),
        unit=str(unit).strip().lower(),
    )
    Hmat = _qml.matrix(H)
    evals = _np.sort(_np.linalg.eigvalsh(_np.asarray(Hmat, dtype=complex)))
    k = int(max(1, k))
    return [float(x) for x in evals[:k]]
