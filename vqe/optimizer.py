import pennylane as qml

OPTIMIZERS = {
    "Adam": qml.AdamOptimizer,
    "GradientDescent": qml.GradientDescentOptimizer,
    "Nesterov": qml.NesterovMomentumOptimizer,
    "Adagrad": qml.AdagradOptimizer,
    "Momentum": qml.MomentumOptimizer,
    "SPSA": qml.SPSAOptimizer,
}

def get_optimizer(name: str, stepsize: float = 0.2):
    """Return PennyLane optimizer by name."""
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' not recognized. Available: {list(OPTIMIZERS.keys())}")
    try:
        return OPTIMIZERS[name](stepsize=stepsize)
    except TypeError:
        return OPTIMIZERS[name](stepsize)

def minimize_energy(circuit, params, optimizer="GradientDescent", lr=0.4, steps=50):
    """Run optimization with the selected optimizer."""
    opt = get_optimizer(optimizer, lr)
    energies = [circuit(params)]
    for n in range(steps):
        try:
            params, _ = opt.step_and_cost(circuit, params)
        except AttributeError:
            params = opt.step(circuit, params)
        energy = circuit(params)
        energies.append(energy)
        print(f"Step {n+1:02d}: E = {energy:.6f}")
    return params, energies
