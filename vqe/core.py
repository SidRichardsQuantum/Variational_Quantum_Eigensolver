import pennylane as qml
from pennylane import numpy as np
from .hamiltonian import build_hamiltonian
from .ansatz import get_ansatz, init_params
from .optimizer import minimize_energy
from .visualize import plot_convergence
from .io_utils import make_run_config_dict, run_signature, save_run_record


def run_vqe(molecule: str = "H2", n_steps: int = 50, plot: bool = True,
            ansatz_name: str = "StronglyEntanglingLayers", optimizer_name: str = "GradientDescent"):
    """Run VQE workflow end-to-end."""
    # --- Build Hamiltonian ---
    H, qubits, symbols, coordinates, basis = build_hamiltonian(molecule)

    # --- Device & ansatz setup ---
    ansatz_fn = get_ansatz(ansatz_name)
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev)
    def circuit(params):
        ansatz_fn(params, wires=range(qubits))
        return qml.expval(H)

    # --- Initialize parameters ---
    np.random.seed(0)
    params = init_params(ansatz_name, qubits)

    # --- Optimization ---
    params, energies = minimize_energy(
        circuit,
        params,
        optimizer_name,
        steps=n_steps
    )

    final_energy = float(energies[-1])

    # --- Optional plotting ---
    if plot:
        plot_convergence(energies, molecule)

    # --- Record config + results ---
    cfg = make_run_config_dict(
        symbols,
        coordinates,
        basis,
        ansatz_name,
        optimizer_name,
        0.4,
        n_steps,
        0
    )
    sig = run_signature(cfg)
    prefix = f"{molecule}_{optimizer_name}_s0__{sig}"

    record = {
        "config": cfg,
        "result": {
            "energy": final_energy,
            "energies": [float(e) for e in energies],
            "steps": n_steps,
        }
    }
    save_run_record(prefix, record)
    print(f"\n💾 Saved run record to results/{prefix}.json\n")

    return {
        "molecule": molecule,
        "energy": final_energy,
        "steps": n_steps,
        "status": "ok",
        "energies": [float(e) for e in energies],
    }
