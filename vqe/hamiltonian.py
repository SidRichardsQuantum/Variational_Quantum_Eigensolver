import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np


def generate_geometry(molecule: str, param_value: float):
    """Return atomic symbols and coordinates for a given geometry parameter."""
    if molecule.upper() == "H2O_ANGLE":
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

    elif molecule.upper() == "H2_BOND":
        bond_length = param_value
        symbols = ["H", "H"]
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, bond_length],
        ])
        return symbols, coordinates
    
    elif molecule.upper() == "LIH_BOND":
        bond_length = param_value
        symbols = ["Li", "H"]
        coordinates = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, bond_length],
        ])
        return symbols, coordinates

    else:
        raise ValueError(f"Unsupported parametric molecule: {molecule}")


def build_hamiltonian(molecule):
    """Return the Hamiltonian, qubit count, and molecular info for a given molecule."""
    # --- Handle parametric molecules (H2O_ANGLE, H2_BOND, etc.) ---
    if "ANGLE" in molecule.upper() or "BOND" in molecule.upper():
        basis = "STO-3G"
        default_param = 104.5 if "H2O" in molecule.upper() else 0.74
        symbols, coordinates = generate_geometry(molecule, default_param)
        H, qubits = qchem.molecular_hamiltonian(symbols, coordinates, basis=basis)
        return H, qubits, symbols, coordinates, basis

    # --- Standard molecules ---
    mol = molecule.upper()

    if mol == "H2":
        symbols = ["H", "H"]
        coordinates = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.7414]])  # Å
        charge = 0
        basis = "sto-3g"

    elif mol == "LIH":
        symbols = ["Li", "H"]
        coordinates = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.6]])  # Å
        charge = 0
        basis = "sto-3g"

    elif mol == "H2O":
        symbols = ["O", "H", "H"]
        coordinates = np.array([[0.000000, 0.000000, 0.000000],
                                [0.758602, 0.000000, 0.504284],
                                [-0.758602, 0.000000, 0.504284]])  # Å
        charge = 0
        basis = "sto-3g"

    elif mol == "H3+":
        symbols = ["H", "H", "H"]
        coordinates = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.872],
                                [0.755, 0.0, 0.436]])  # Å
        charge = +1
        basis = "sto-3g"

    else:
        raise ValueError(
            f"Unsupported molecule '{molecule}'. "
            "Available options: H2, LiH, H2O, H3+"
        )

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        basis=basis,
        unit="angstrom",
    )

    return H, qubits, symbols, coordinates, basis
