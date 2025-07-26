import pennylane as qml
import numpy as np
# import matplotlib.pyplot as plt


# Define the Ansäzes
def uccsd_ansatz(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(-params[0], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def ry_cz_ansatz(params, wires):
    for i in range(len(wires)):
        qml.RY(params[i], wires=wires[i])
    for i in range(len(wires) - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])


def minimal_ansatz(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    
# Create list of ansätzes
ANSATZES = {
    "UCCSD": uccsd_ansatz,
    "RY-CZ": ry_cz_ansatz,
    "Minimal": minimal_ansatz,
}

# Create list of optimizers
OPTIMIZERS = {
        "Adam": qml.AdamOptimizer,
        "GradientDescent": qml.GradientDescentOptimizer,
        "Nesterov": qml.NesterovMomentumOptimizer,
        "Adagrad": qml.AdagradOptimizer,
        "Momentum": qml.MomentumOptimizer,
        "SPSA": qml.SPSAOptimizer,
    }


def get_optimizer(name: str, stepsize: float = 0.2):
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' not recognized.")
    return OPTIMIZERS[name](stepsize)


# Define VQE circuit function
def create_vqe_circuit(ansatz_fn, hamiltonian, dev, wires):
    @qml.qnode(dev)
    def circuit(params):
        ansatz_fn(params, wires=wires)
        return qml.expval(hamiltonian)
    return circuit


# Initial parameter function
def init_params(ansatz_name, num_wires):
    if ansatz_name in ["UCCSD", "Minimal"]:
        return 0.01 * np.random.randn(1)
    elif ansatz_name == "RY-CZ":
        return 0.01 * np.random.randn(num_wires)
    else:
        raise ValueError("Unknown ansatz")


# Function that runs the VQE algorithm for a given optimizer and ansätz
def run_vqe(cost_fn, init_params, optimizer, stepsize=0.2, max_iters=50):
    opt = optimizer(stepsize) if isinstance(stepsize, float) else optimizer()
    params = init_params
    energies = []

    for _ in range(max_iters):
        params = opt.step(cost_fn, params)
        energy = cost_fn(params)
        energies.append(energy)

    return params, energies


def excitation_ansatz(params, wires, hf_state, excitations, excitation_type="both"):
    qml.BasisState(hf_state, wires=wires)

    if excitation_type == "double":
        for i, exc in enumerate(excitations):
            qml.DoubleExcitation(params[i], wires=exc)
    elif excitation_type == "single":
        for i, exc in enumerate(excitations):
            qml.SingleExcitation(params[i], wires=exc)
    elif excitation_type == "both":
        singles, doubles = excitations
        for i, exc in enumerate(singles):
            qml.SingleExcitation(params[i], wires=exc)
        for j, exc in enumerate(doubles):
            qml.DoubleExcitation(params[len(singles) + j], wires=exc)
    else:
        raise ValueError("Invalid excitation_type")
