import pennylane as qml
from pennylane import numpy as np
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
    "TwoQubit-RY-CNOT": uccsd_ansatz,
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
def init_params_for(ansatz_name, num_wires):
    if ansatz_name in ["TwoQubit-RY-CNOT", "Minimal"]:
        return 0.01 * np.random.randn(1, requires_grad=True)
    elif ansatz_name == "RY-CZ":
        return 0.01 * np.random.randn(num_wires, requires_grad=True)
    raise ValueError("Unknown ansatz")


def _normalize_optimizer(opt_like, stepsize: float):
    # accept name | class | instance
    if isinstance(opt_like, str):
        return get_optimizer(opt_like, stepsize)
    if isinstance(opt_like, type):
        return opt_like(stepsize=stepsize)
    return opt_like


def run_vqe(cost_fn, initial_params, optimizer="Adam", stepsize=0.2, max_iters=50):
    opt = _normalize_optimizer(optimizer, stepsize)
    params = np.array(initial_params, requires_grad=True)
    energies = []
    for _ in range(max_iters):
        params = opt.step(cost_fn, params)
        energies.append(cost_fn(params))
    return params, energies


def excitation_ansatz(params, wires, hf_state, excitations, excitation_type="both"):
    qml.BasisState(np.array(hf_state, dtype=int), wires=wires)

    # flat list mode for backward compatibility
    if excitation_type in ["single", "double"] and isinstance(excitations, list):
        if excitation_type == "single":
            if len(params) != len(excitations):
                raise ValueError("params length must match number of single excitations")
            for i, exc in enumerate(excitations):
                qml.SingleExcitation(params[i], wires=exc)
        else:
            if len(params) != len(excitations):
                raise ValueError("params length must match number of double excitations")
            for i, exc in enumerate(excitations):
                qml.DoubleExcitation(params[i], wires=exc)
        return

    # tuple mode: (singles, doubles)
    singles, doubles = excitations
    needed = (len(singles) if excitation_type in ["single", "both"] else 0) + \
             (len(doubles) if excitation_type in ["double", "both"] else 0)
    if len(params) != needed:
        raise ValueError(f"params length {len(params)} != required {needed}")

    i = 0
    if excitation_type in ["single", "both"]:
        for exc in singles:
            qml.SingleExcitation(params[i], wires=exc); i += 1
    if excitation_type in ["double", "both"]:
        for exc in doubles:
            qml.DoubleExcitation(params[i], wires=exc); i += 1
