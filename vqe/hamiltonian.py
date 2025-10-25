import pennylane as qml
from pennylane import numpy as np

def build_hamiltonian(molecule: str = "H2"):
    """Return Hamiltonian and number of qubits for a simple molecule."""
    if molecule.upper() == "H2":
        symbols = ["H", "H"]
        coordinates = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.7414]])  # 0.7414 Å
        charge = 0
        basis_set = "sto-3g"
    else:
        raise ValueError(f"Unsupported molecule: {molecule}")

    H, qubits = qml.qchem.molecular_hamiltonian(
        symbols,
        coordinates,
        charge=charge,
        basis=basis_set,
        unit="angstrom"
    )

    return H, qubits
