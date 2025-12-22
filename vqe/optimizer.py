"""
vqe.optimizer
-------------
Lightweight wrapper over PennyLane optimizers with a unified interface.

Provides:
    - get_optimizer(name, stepsize)
"""

from __future__ import annotations

import pennylane as qml

# ================================================================
# AVAILABLE OPTIMIZERS
# ================================================================
_OPTIMIZERS = {
    "Adam": qml.AdamOptimizer,
    "adam": qml.AdamOptimizer,  # alias
    "GradientDescent": qml.GradientDescentOptimizer,
    "gd": qml.GradientDescentOptimizer,  # alias
    "Momentum": qml.MomentumOptimizer,
    "Nesterov": qml.NesterovMomentumOptimizer,
    "RMSProp": qml.RMSPropOptimizer,
    "Adagrad": qml.AdagradOptimizer,
}


# ================================================================
# MAIN FACTORY
# ================================================================
def get_optimizer(name: str = "Adam", stepsize: float = 0.2):
    """
    Return a PennyLane optimizer instance by name.

    Args:
        name: Optimizer identifier (case-insensitive).
        stepsize: Learning rate.

    Returns:
        An instantiated optimizer.
    """
    key = name.lower()
    for k, cls in _OPTIMIZERS.items():
        if k.lower() == key:
            return cls(stepsize)

    raise ValueError(
        f"Unknown optimizer '{name}'. " f"Available: {', '.join(_OPTIMIZERS.keys())}"
    )
