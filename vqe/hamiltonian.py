import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np


# ================================================================
# GEOMETRY GENERATORS
# ================================================================
def generate_geometry(molecule: str, param_value: float):
    """Return atomic symbols and coordinates for a given geometry parameter."""
    mol = molecule.upper()

    if mol == "H2O_ANGLE":
        bond_length = 0.9584  # in Å
        angle_deg = param_value
        angle_rad = np.deg2rad(angle_deg)
        x = bond_length * np.sin(angle_rad / 2)
        z = bond_length * np.cos(angle_rad / 2)
        symbols = ["O", "H", "H"]
        coordinates = np.array([
            [0.0, 0.0, 0.0],   # Oxygen
            [x, 0.0, z],       # Hydrogen 1
            [-x, 0.0, z],      # Hydrogen 2
        ])
        return symbols, coordinates

    elif mol == "H2_BOND":
        symbols = ["H", "H"]
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, param_value],
        ])
        return symbols, coordinates

    elif mol == "LIH_BOND":
        symbols = ["Li", "H"]
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, param_value],
        ])
        return symbols, coordinates

    else:
        raise ValueError(f"Unsupported parametric molecule: {molecule}")


# ================================================================
# HAMILTONIAN BUILDER
# ================================================================
def build_hamiltonian(molecule: str, mapping: str = "jordan_wigner"):
    """Return the Hamiltonian, qubit count, and molecular info for a given molecule.

    Args:
        molecule (str): Molecule name (e.g., 'H2', 'LiH', 'H2O', or parametric forms like 'H2O_ANGLE').
        mapping (str): Fermion-to-qubit mapping. One of:
            'jordan_wigner', 'bravyi_kitaev', 'parity'.
    """
    mol = molecule.upper()

    # --- Parametric molecules (for geometry scans) ---
    if "ANGLE" in mol or "BOND" in mol:
        basis = "sto-3g"
        default_param = 104.5 if "H2O" in mol else 0.74
        symbols, coordinates = generate_geometry(molecule, default_param)
        charge = 0
    else:
        # --- Static molecules ---
        if mol == "H2":
            symbols = ["H", "H"]
            coordinates = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.7414],
            ])
            charge = 0
            basis = "sto-3g"

        elif mol == "LIH":
            symbols = ["Li", "H"]
            coordinates = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.6],
            ])
            charge = 0
            basis = "sto-3g"

        elif mol == "H2O":
            symbols = ["O", "H", "H"]
            coordinates = np.array([
                [0.000000, 0.000000, 0.000000],
                [0.758602, 0.000000, 0.504284],
                [-0.758602, 0.000000, 0.504284],
            ])
            charge = 0
            basis = "sto-3g"

        elif mol == "H3+":
            symbols = ["H", "H", "H"]
            coordinates = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.872],
                [0.755, 0.0, 0.436],
            ])
            charge = +1
            basis = "sto-3g"

        else:
            raise ValueError(
                f"Unsupported molecule '{molecule}'. "
                "Available options: H2, LiH, H2O, H3+, H2_BOND, H2O_ANGLE, LiH_BOND"
            )

    # --- Build Hamiltonian using the requested mapping ---
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
        # Older PennyLane versions (<0.35) may not support mapping kwarg
        print(f"⚠️ Mapping '{mapping}' not supported in this PennyLane version — using Jordan–Wigner.")
        H, qubits = qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=charge,
            basis=basis,
            unit="angstrom",
        )

    return H, qubits, symbols, coordinates, basis
