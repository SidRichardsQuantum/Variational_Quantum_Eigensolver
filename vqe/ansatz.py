import pennylane as qml
from pennylane import numpy as np

def two_qubit_ry_cnot(params, wires):
    """Toy 2-qubit entangler; NOT chemical UCCSD."""
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(-params[0], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

def ry_cz(params, wires):
    """RY rotations followed by CZ entanglement."""
    if len(params) != len(wires):
        raise ValueError("ry_cz expects one parameter per wire")
    for w in range(len(wires)):
        qml.RY(params[w], wires=wires[w])
    for i in range(len(wires) - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])

def minimal(params, wires):
    """Minimal 2-qubit circuit."""
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])

def hardware_efficient_ansatz(params, wires):
    """Default hardware-efficient ansatz."""
    qml.templates.StronglyEntanglingLayers(params, wires=wires)

ANSATZES = {
    "TwoQubit-RY-CNOT": two_qubit_ry_cnot,
    "RY-CZ": ry_cz,
    "Minimal": minimal,
    "StronglyEntanglingLayers": hardware_efficient_ansatz,
}

def get_ansatz(name: str):
    """Return ansatz function by name."""
    if name not in ANSATZES:
        raise ValueError(f"Ansatz '{name}' not recognized. Available: {list(ANSATZES.keys())}")
    return ANSATZES[name]

def init_params(ansatz_name: str, num_wires: int, scale: float = 0.01, requires_grad=True):
    """Initialize parameters based on ansatz structure."""
    if ansatz_name in ["TwoQubit-RY-CNOT", "Minimal"]:
        vals = scale * np.random.randn(1)
    elif ansatz_name == "RY-CZ":
        vals = scale * np.random.randn(num_wires)
    elif ansatz_name == "StronglyEntanglingLayers":
        # Use a small but reasonable initialization
        vals = np.random.normal(0, np.pi, (1, num_wires, 3))
    else:
        raise ValueError(f"Unknown ansatz '{ansatz_name}'")
    return np.array(vals, requires_grad=requires_grad)
