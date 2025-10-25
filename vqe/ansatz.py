import pennylane as qml

def hardware_efficient_ansatz(params, wires):
    """Define a simple hardware-efficient ansatz."""
    qml.templates.StronglyEntanglingLayers(params, wires=wires)
