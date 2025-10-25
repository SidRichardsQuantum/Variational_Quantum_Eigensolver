import pennylane as qml
from pennylane import numpy as np

def build_hamiltonian(molecule: str = "H2"):
    """Return Hamiltonian and qubit count for a simple molecule."""
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
        # Equilateral triangle, side ≈ 1.65 Bohr ≈ 0.872 Å
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
