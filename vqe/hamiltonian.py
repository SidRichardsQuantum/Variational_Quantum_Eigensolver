"""
vqe.hamiltonian
---------------
Molecular Hamiltonian and geometry utilities for VQE simulations.

Provides:
- `generate_geometry`: Parametric molecule generators for bond lengths and angles.
- `build_hamiltonian`: Construction of fermionic Hamiltonians mapped to qubit operators.
"""

import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np


# ================================================================
# GEOMETRY GENERATORS
# ================================================================
def generate_geometry(molecule: str, param_value: float):
    """
    Generate atomic symbols and coordinates for a parameterized molecule.

    Supported forms:
        - "H2_BOND": Varies H–H bond length
        - "LIH_BOND": Varies Li–H bond length
        - "H2O_ANGLE": Varies H–O–H bond angle (°)

    Args:
        molecule: Molecule identifier (case-insensitive)
        param_value: Geometry parameter (bond length in Å or angle in degrees)

    Returns:
        (symbols, coordinates): Lists and arrays defining the molecule.
    """
    mol = molecule.upper()

    if mol == "H2O_ANGLE":
        bond_length = 0.9584  # in Å
        angle_rad = np.deg2rad(param_value)
        x = bond_length * np.sin(angle_rad / 2)
        z = bond_length * np.cos(angle_rad / 2)
        symbols = ["O", "H", "H"]
        coordinates = np.array([
            [0.0, 0.0, 0.0],  # Oxygen
            [x, 0.0, z],      # Hydrogen 1
            [-x, 0.0, z],     # Hydrogen 2
        ])
        return symbols, coordinates

    if mol == "H2_BOND":
        symbols = ["H", "H"]
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, param_value],
        ])
        return symbols, coordinates

    if mol == "LIH_BOND":
        symbols = ["Li", "H"]
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, param_value],
        ])
        return symbols, coordinates

    raise ValueError(
        f"Unsupported parametric molecule '{molecule}'. "
        "Supported: H2O_ANGLE, H2_BOND, LiH_BOND."
    )


# ================================================================
# HAMILTONIAN BUILDER
# ================================================================
def build_hamiltonian(molecule: str, mapping: str = "jordan_wigner"):
    """
    Construct the qubit Hamiltonian for a given molecule using PennyLane's qchem.

    Supports static and parameterized molecules.

    Args:
        molecule: Molecule name (e.g., "H2", "LiH", "H2O", "H3+")
                  or parametric variants ("H2_BOND", "H2O_ANGLE", "LiH_BOND").
        mapping: Fermion-to-qubit mapping scheme.
                 One of {"jordan_wigner", "bravyi_kitaev", "parity"}.

    Returns:
        tuple: (H, num_qubits, symbols, coordinates, basis)
            - H: Qubit Hamiltonian (qml.Hamiltonian)
            - num_qubits: Number of qubits required
            - symbols: List of atomic symbols
            - coordinates: Molecular geometry array
            - basis: Basis set used
    """
    mol = molecule.upper()

    # ------------------------------------------------------------
    # Handle parameterized molecules (geometry scans)
    # ------------------------------------------------------------
    if "ANGLE" in mol or "BOND" in mol:
        basis = "sto-3g"
        default_param = 104.5 if "H2O" in mol else 0.74
        symbols, coordinates = generate_geometry(molecule, default_param)
        charge = 0

    # ------------------------------------------------------------
    # Handle static molecules
    # ------------------------------------------------------------
    else:
        if mol == "H2":
            symbols = ["H", "H"]
            coordinates = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.7414],
            ])
            charge, basis = 0, "sto-3g"

        elif mol == "LIH":
            symbols = ["Li", "H"]
            coordinates = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.6],
            ])
            charge, basis = 0, "sto-3g"

        elif mol == "H2O":
            symbols = ["O", "H", "H"]
            coordinates = np.array([
                [0.000000, 0.000000, 0.000000],
                [0.758602, 0.000000, 0.504284],
                [-0.758602, 0.000000, 0.504284],
            ])
            charge, basis = 0, "sto-3g"

        elif mol == "H3+":
            symbols = ["H", "H", "H"]
            coordinates = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.872],
                [0.755, 0.0, 0.436],
            ])
            charge, basis = +1, "sto-3g"

        else:
            raise ValueError(
                f"Unsupported molecule '{molecule}'. "
                "Available: H2, LiH, H2O, H3+, H2_BOND, H2O_ANGLE, LiH_BOND."
            )

    # ------------------------------------------------------------
    # Build molecular Hamiltonian
    # ------------------------------------------------------------
    try:
        H, qubits = qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=charge,
            basis=basis,
            mapping=mapping,
            unit="angstrom",
        )
    except TypeError:
        # Fallback for older PennyLane versions that lack `mapping`
        print(f"⚠️  Mapping '{mapping}' not supported in this PennyLane version — defaulting to Jordan–Wigner.")
        H, qubits = qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=charge,
            basis=basis,
            unit="angstrom",
        )

    return H, qubits, symbols, coordinates, basis
