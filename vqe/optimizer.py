from pennylane import numpy as np
import pennylane as qml

def minimize_energy(circuit, params, steps: int = 50, lr: float = 0.4):
    """Run gradient descent on the given circuit."""
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    energies = [circuit(params)]

    for n in range(steps):
        params = opt.step(circuit, params)
        energy = circuit(params)
        energies.append(energy)
        print(f"Step {n+1:02d}: E = {energy:.6f}")

    return params, energies
