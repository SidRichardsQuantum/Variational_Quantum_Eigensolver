import pennylane as qml
from pennylane import numpy as np
import json, hashlib, os, glob
from typing import Any, Dict


# Directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "vqe")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
IMG_DIR = os.path.join(DATA_DIR, "images")

# Ensure required directories exist
def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)

# Helper to round floats for stable hashing
def _round_floats(x: Any, ndigits: int = 8):
    """Recursively round floats / arrays / lists for stable hashing."""
    if isinstance(x, float):
        return round(x, ndigits)
    try:
        if hasattr(x, "item") and isinstance(x.item(), float):
            return round(float(x), ndigits)
    except Exception:
        pass
    if hasattr(x, "tolist"):
        return _round_floats(x.tolist(), ndigits)
    if isinstance(x, (list, tuple)):
        return type(x)(_round_floats(v, ndigits) for v in x)
    return x

# Config dict for a VQE run
def make_run_config_dict(
    symbols,
    coordinates,
    basis: str,
    ansatz_desc: str,
    optimizer_name: str,
    stepsize: float,
    max_iterations: int,
    seed: int,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
) -> Dict[str, Any]:
    """Canonical config for hashing & cache keys, now includes noise info."""
    return {
        "symbols": list(symbols),
        "geometry": _round_floats(coordinates, 8),
        "basis": basis,
        "ansatz": ansatz_desc,
        "optimizer": {
            "name": optimizer_name,
            "stepsize": float(stepsize),
            "iterations_planned": int(max_iterations),
        },
        "seed": int(seed),
        "noisy": noisy,
        "depolarizing_prob": float(depolarizing_prob),
        "amplitude_damping_prob": float(amplitude_damping_prob),
    }

# Run signature
def run_signature(cfg: Dict[str, Any]) -> str:
    """Stable short hash of the config (12 hex chars)."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

# Find existing run
def find_existing_run(sig):
    """Look for an existing results file matching the signature."""
    pattern_results = os.path.join(RESULTS_DIR, f"*__{sig}.json")
    matches = sorted(glob.glob(pattern_results))
    return matches[-1] if matches else None

# Save run record
def save_run_record(record_name_prefix, record):
    """Save run record directly into results/."""
    ensure_dirs()
    fname_results = os.path.join(RESULTS_DIR, f"{record_name_prefix}.json")
    with open(fname_results, "w") as f:
        json.dump(record, f, indent=2)
    return fname_results

# File naming
def build_run_filename(prefix: str, optimizer_name: str, seed: int, sig: str) -> str:
    """Consistent file naming helper, e.g. H2_noiseless_Adam_s0__abc123def456.json"""
    safe_opt = optimizer_name.replace(" ", "")
    return f"{prefix}_{safe_opt}_s{seed}__{sig}"

# Ansatz definitions
def two_qubit_ry_cnot(params, wires):
    """Toy 2-qubit entangler; NOT chemical UCCSD."""
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(-params[0], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])

def ry_cz(params, wires):
    if len(params) != len(wires):
        raise ValueError("ry_cz expects one parameter per wire")
    for w in wires:
        qml.RY(params[w], wires=w)
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

# Optimizer retrieval
def get_optimizer(name: str, stepsize: float = 0.2):
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' not recognized.")
    try:
        return OPTIMIZERS[name](stepsize=stepsize)
    except TypeError:
        return OPTIMIZERS[name](stepsize)

# Parameter initialization
def init_params(ansatz_name: str, num_wires: int, scale=0.01, requires_grad=True):
    if ansatz_name in ["TwoQubit-RY-CNOT", "Minimal"]:
        vals = scale * np.random.randn(1)
    elif ansatz_name == "RY-CZ":
        vals = scale * np.random.randn(num_wires)
    else:
        raise ValueError(f"Unknown ansatz '{ansatz_name}'")
    return np.array(vals, requires_grad=requires_grad)

# VQE circuit creation
def create_vqe_circuit(ansatz_fn, hamiltonian, dev, wires):
    @qml.qnode(dev)
    def circuit(params):
        ansatz_fn(params, wires=wires)
        return qml.expval(hamiltonian)
    return circuit

# Normalize optimizer input
def _normalize_optimizer(opt_like, stepsize: float):
    if isinstance(opt_like, str):
        return get_optimizer(opt_like, stepsize)
    if isinstance(opt_like, type):
        try:
            return opt_like(stepsize=stepsize)
        except TypeError:
            return opt_like(stepsize)
    return opt_like

# Run VQE
def run_vqe(cost_fn, initial_params, optimizer="Adam", stepsize=0.2, max_iters=50, record_initial=True):
    opt = _normalize_optimizer(optimizer, stepsize)
    params = np.array(initial_params, requires_grad=True)
    energies = [cost_fn(params)] if record_initial else []
    for _ in range(max_iters):
        try:
            params, _ = opt.step_and_cost(cost_fn, params)
        except AttributeError:
            params = opt.step(cost_fn, params)
        energies.append(cost_fn(params))
    return params, energies

# Set random seed
def set_seed(seed=0):
    np.random.seed(seed)

# Excitation ansatz
def excitation_ansatz(params, wires, hf_state, excitations, excitation_type="both"):
    qml.BasisState(np.array(hf_state, dtype=int), wires=wires)
    if excitation_type in ["single", "double"] and isinstance(excitations, list):
        ex_list = excitations
        if len(params) != len(ex_list):
            raise ValueError("params length must match number of excitations")
        gate = qml.SingleExcitation if excitation_type == "single" else qml.DoubleExcitation
        for i, exc in enumerate(ex_list):
            gate(params[i], wires=exc)
        return
    singles, doubles = excitations
    n_required = (len(singles) if excitation_type in ["single", "both"] else 0) + \
                 (len(doubles) if excitation_type in ["double", "both"] else 0)
    if len(params) != n_required:
        raise ValueError(f"params length {len(params)} != required {n_required}")
    i = 0
    if excitation_type in ["single", "both"]:
        for exc in singles:
            qml.SingleExcitation(params[i], wires=exc)
            i += 1
    if excitation_type in ["double", "both"]:
        for exc in doubles:
            qml.DoubleExcitation(params[i], wires=exc)
            i += 1
