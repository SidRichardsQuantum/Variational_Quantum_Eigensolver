import pennylane as qml
from pennylane import numpy as np

# ---- Ansatzes ----
def two_qubit_ry_cnot(params, wires):
    """Toy 2-qubit entangler; NOT chemical UCCSD."""
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(-params[0], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

def ry_cz(params, wires):
    if len(params) != len(wires):
        raise ValueError("ry_cz expects one parameter per wire")
    for w in wires: qml.RY(params[w], wires=w)
    for i in range(len(wires) - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])

def minimal(params, wires):
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])

ANSATZES = {
    "TwoQubit-RY-CNOT": two_qubit_ry_cnot,
    "RY-CZ": ry_cz,
    "Minimal": minimal,
}

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
    try: return OPTIMIZERS[name](stepsize=stepsize)
    except TypeError: return OPTIMIZERS[name](stepsize)

def init_params(ansatz_name: str, num_wires: int, scale=0.01, requires_grad=True):
    if ansatz_name in ["TwoQubit-RY-CNOT", "Minimal"]:
        vals = scale * np.random.randn(1)
    elif ansatz_name == "RY-CZ":
        vals = scale * np.random.randn(num_wires)
    else:
        raise ValueError(f"Unknown ansatz '{ansatz_name}'")
    return np.array(vals, requires_grad=requires_grad)

def create_vqe_circuit(ansatz_fn, hamiltonian, dev, wires):
    @qml.qnode(dev)
    def circuit(params):
        ansatz_fn(params, wires=wires)
        return qml.expval(hamiltonian)
    return circuit

def _normalize_optimizer(opt_like, stepsize: float):
    if isinstance(opt_like, str): return get_optimizer(opt_like, stepsize)
    if isinstance(opt_like, type):
        try: return opt_like(stepsize=stepsize)
        except TypeError: return opt_like(stepsize)
    return opt_like

def run_vqe(cost_fn, initial_params, optimizer="Adam", stepsize=0.2, max_iters=50, record_initial=True):
    opt = _normalize_optimizer(optimizer, stepsize)
    params = np.array(initial_params, requires_grad=True)
    energies = [cost_fn(params)] if record_initial else []
    for _ in range(max_iters):
        try: params, _ = opt.step_and_cost(cost_fn, params)
        except AttributeError: params = opt.step(cost_fn, params)
        energies.append(cost_fn(params))
    return params, energies

def set_seed(seed=0):
    np.random.seed(seed)

def excitation_ansatz(params, wires, hf_state, excitations, excitation_type="both"):
    qml.BasisState(np.array(hf_state, dtype=int), wires=wires)
    if excitation_type in ["single", "double"] and isinstance(excitations, list):
        ex_list = excitations
        if len(params) != len(ex_list):
            raise ValueError("params length must match number of excitations")
        gate = qml.SingleExcitation if excitation_type == "single" else qml.DoubleExcitation
        for i, exc in enumerate(ex_list): gate(params[i], wires=exc)
        return
    singles, doubles = excitations
    n_required = (len(singles) if excitation_type in ["single", "both"] else 0) + \
                 (len(doubles) if excitation_type in ["double", "both"] else 0)
    if len(params) != n_required:
        raise ValueError(f"params length {len(params)} != required {n_required}")
    i = 0
    if excitation_type in ["single", "both"]:
        for exc in singles: qml.SingleExcitation(params[i], wires=exc); i += 1
    if excitation_type in ["double", "both"]:
        for exc in doubles: qml.DoubleExcitation(params[i], wires=exc); i += 1
