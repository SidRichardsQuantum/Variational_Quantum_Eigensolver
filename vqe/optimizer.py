"""
vqe.optimizer
-------------
Lightweight interface for selecting and running PennyLane optimizers.
"""

import pennylane as qml

# ================================================================
# AVAILABLE OPTIMIZERS
# ================================================================
OPTIMIZERS = {
    "Adam": qml.AdamOptimizer,
    "GradientDescent": qml.GradientDescentOptimizer,
    "Nesterov": qml.NesterovMomentumOptimizer,
    "Adagrad": qml.AdagradOptimizer,
    "Momentum": qml.MomentumOptimizer,
    "SPSA": qml.SPSAOptimizer,
}


# ================================================================
# OPTIMIZER SELECTION
# ================================================================
def get_optimizer(name: str, stepsize: float = 0.2):
    """
    Retrieve a PennyLane optimizer instance by name.

    Args:
        name: Optimizer name (e.g., "Adam", "GradientDescent").
        stepsize: Step size (learning rate) to initialize the optimizer.

    Returns:
        An instantiated PennyLane optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    if name not in OPTIMIZERS:
        available = ", ".join(OPTIMIZERS.keys())
        raise ValueError(f"Unknown optimizer '{name}'. Available options: {available}")

    # Handle both modern (stepsize=kwarg) and legacy (positional) API variants
    try:
        return OPTIMIZERS[name](stepsize=stepsize)
    except TypeError:
        return OPTIMIZERS[name](stepsize)


# ================================================================
# OPTIMIZATION LOOP
# ================================================================
def minimize_energy(circuit, params, optimizer: str = "Adam", steps: int = 50, stepsize: float = 0.2):
    """
    Run iterative optimization of a variational circuit to minimize energy.

    Args:
        circuit: A callable QNode returning the circuit energy.
        params: Initial parameter array.
        optimizer: Optimizer name (default: "Adam").
        steps: Number of optimization steps.
        stepsize: Step size for the optimizer.

    Returns:
        (final_params, energies)
            final_params: Optimized parameters.
            energies: List of energy values per iteration (including initial).
    """
    opt = get_optimizer(optimizer, stepsize)

    energies = [circuit(params)]
    for n in range(steps):
        try:
            params, _ = opt.step_and_cost(circuit, params)
        except AttributeError:
            # For optimizers that lack step_and_cost
            params = opt.step(circuit, params)

        energy = circuit(params)
        energies.append(energy)
        print(f"Step {n + 1:02d}/{steps}: E = {energy:.6f}")

    return params, energies
