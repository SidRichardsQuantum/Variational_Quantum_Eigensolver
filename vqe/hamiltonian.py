"""
vqe.hamiltonian
---------------
Molecular Hamiltonian and geometry utilities for VQE simulations.

Provides:
- A registry of static molecular presets (`MOLECULES`)
- `generate_geometry`: parametric generators for bond lengths / angles
- `build_hamiltonian`: construction of qubit Hamiltonians with mappings
"""

from __future__ import annotations

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np


# ================================================================
# STATIC MOLECULE REGISTRY
# ================================================================

#: Canonical presets for common molecules used across the project.
#:
#: Each entry has:
#:   - symbols: list[str]      → atomic species
#:   - coordinates: np.ndarray → shape (N, 3), in Å
#:   - charge: int             → total molecular charge
#:   - basis: str              → basis set name (string used in JSON/configs)
MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.7414],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
    },
    "LIH": {
        "symbols": ["Li", "H"],
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.6],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
    },
    "H2O": {
        "symbols": ["O", "H", "H"],
        "coordinates": np.array(
            [
                [0.000000, 0.000000, 0.000000],
                [0.758602, 0.000000, 0.504284],
                [-0.758602, 0.000000, 0.504284],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
    },
    # Canonical H3+ geometry: equilateral triangle in the xy-plane
    "H3+": {
        "symbols": ["H", "H", "H"],
        "coordinates": np.array(
            [
                [0.000000, 1.000000, 0.000000],
                [-0.866025, -0.500000, 0.000000],
                [0.866025, -0.500000, 0.000000],
            ]
        ),
        "charge": +1,
        "basis": "STO-3G",
    },
}


# ================================================================
# PARAMETRIC GEOMETRY GENERATORS
# ================================================================
def generate_geometry(molecule: str, param_value: float):
    """
    Generate atomic symbols and coordinates for a parameterised molecule.

    Supported identifiers (case-insensitive):
        - "H2_BOND"   : varies the H–H bond length (Å)
        - "LIH_BOND"  : varies the Li–H bond length (Å)
        - "H2O_ANGLE" : varies the H–O–H bond angle (degrees)

    Args:
        molecule: Molecule identifier including the parameter tag,
                  e.g. "H2_BOND", "LiH_BOND", "H2O_ANGLE".
        param_value: Geometry parameter:
                     - bond length in Å for *_BOND
                     - angle in degrees for *_ANGLE

    Returns:
        (symbols, coordinates):
            - symbols: list[str]
            - coordinates: np.ndarray of shape (N, 3), in Å
    """
    mol = molecule.upper()

    if mol == "H2O_ANGLE":
        # Water with fixed bond length, variable angle
        bond_length = 0.9584  # Å
        angle_rad = np.deg2rad(param_value)
        x = bond_length * np.sin(angle_rad / 2)
        z = bond_length * np.cos(angle_rad / 2)

        symbols = ["O", "H", "H"]
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],  # Oxygen
                [x, 0.0, z],      # Hydrogen 1
                [-x, 0.0, z],     # Hydrogen 2
            ]
        )
        return symbols, coordinates

    if mol == "H2_BOND":
        # Dihydrogen with variable bond length along z
        symbols = ["H", "H"]
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, float(param_value)],
            ]
        )
        return symbols, coordinates

    if mol == "LIH_BOND":
        # Lithium hydride with variable Li–H separation along z
        symbols = ["Li", "H"]
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, float(param_value)],
            ]
        )
        return symbols, coordinates

    raise ValueError(
        f"Unsupported parametric molecule '{molecule}'. "
        "Supported: H2_BOND, LiH_BOND, H2O_ANGLE."
    )


# ================================================================
# HAMILTONIAN BUILDER
# ================================================================
def _get_preset(molecule: str):
    """Internal helper: fetch preset entry from MOLECULES by name (case-insensitive)."""
    key = molecule.upper()
    if key in MOLECULES:
        return MOLECULES[key]

    # Handle "H3PLUS" alias for convenience
    if key in {"H3PLUS", "H3_PLUS"}:
        return MOLECULES["H3+"]

    available = ", ".join(MOLECULES.keys())
    raise ValueError(
        f"Unsupported molecule '{molecule}'. "
        f"Available static presets: {available}, or parametric: H2_BOND, LiH_BOND, H2O_ANGLE."
    )


def build_hamiltonian(molecule: str, mapping: str = "jordan_wigner"):
    """
    Construct the qubit Hamiltonian for a given molecule using PennyLane's qchem.

    This is the **single source of truth** for Hamiltonian construction
    in the VQE/SSVQE workflows.

    Supports:
        - Static presets:
            "H2", "LiH", "H2O", "H3+"
        - Parametric variants:
            "H2_BOND", "LiH_BOND", "H2O_ANGLE"

    Args:
        molecule:
            Molecule identifier (case-insensitive). Examples:
            - "H2", "LiH", "H2O", "H3+"
            - "H2_BOND", "LiH_BOND", "H2O_ANGLE"
        mapping:
            Fermion-to-qubit mapping scheme.
            One of {"jordan_wigner", "bravyi_kitaev", "parity"}.
            Case-insensitive; stored in the config as lower-case.

    Returns:
        (H, num_qubits, symbols, coordinates, basis)
            - H: qml.Hamiltonian
            - num_qubits: int
            - symbols: list[str]
            - coordinates: np.ndarray, in Å
            - basis: str (e.g. "STO-3G")
    """
    mapping = mapping.lower()
    mol = molecule.upper()

    # ------------------------------------------------------------
    # Parametric molecules: delegate to generator with defaults
    # ------------------------------------------------------------
    if "BOND" in mol or "ANGLE" in mol:
        # - H2_BOND, LiH_BOND: bond length default is ~0.74–1.0 Å,
        #                      but actual scans always call generate_geometry
        # - H2O_ANGLE: default around 104.5°
        if mol == "H2O_ANGLE":
            default_param = 104.5  # degrees
        else:
            default_param = 0.74   # Å; purely a placeholder

        symbols, coordinates = generate_geometry(molecule, default_param)
        charge = 0
        basis = "STO-3G"

    # ------------------------------------------------------------
    # Static presets from the registry
    # ------------------------------------------------------------
    else:
        preset = _get_preset(mol)
        symbols = preset["symbols"]
        coordinates = preset["coordinates"]
        charge = preset["charge"]
        basis = preset["basis"]

    # ------------------------------------------------------------
    # Build molecular Hamiltonian with mapping
    # ------------------------------------------------------------
    try:
        H, num_qubits = qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=charge,
            basis=basis,
            mapping=mapping,
            unit="angstrom",
        )
    except TypeError:
        # Fallback for older PennyLane versions that do not support `mapping=`
        print(
            f"⚠️  Mapping '{mapping}' not supported in this PennyLane version — "
            "defaulting to Jordan–Wigner."
        )
        H, num_qubits = qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=charge,
            basis=basis,
            unit="angstrom",
        )

    return H, num_qubits, symbols, coordinates, basis
