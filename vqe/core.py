import pennylane as qml
from pennylane import numpy as np
from .hamiltonian import build_hamiltonian
from .ansatz import hardware_efficient_ansatz
from .optimizer import minimize_energy
from .visualize import plot_convergence

def run_vqe(molecule: str = "H2", n_steps: int = 50, plot: bool = True):
    """Run VQE workflow end-to-end."""
    H, qubits = build_hamiltonian(molecule)
    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev)
    def circuit(params):
        hardware_efficient_ansatz(params, wires=range(qubits))
        return qml.expval(H)

    np.random.seed(0)
    params = np.random.normal(0, np.pi, (1, qubits, 3))
    params, energies = minimize_energy(circuit, params, steps=n_steps)

    final_energy = float(energies[-1])
    if plot:
        plot_convergence(energies, molecule)

    return {
        "molecule": molecule,
        "energy": final_energy,
        "steps": n_steps,
        "status": "ok",
        "energies": energies,
    }
