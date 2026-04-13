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
OPTIMIZERS = {
    "Adam": qml.AdamOptimizer,
    "GradientDescent": qml.GradientDescentOptimizer,
    "Momentum": qml.MomentumOptimizer,
    "NesterovMomentum": qml.NesterovMomentumOptimizer,
    "RMSProp": qml.RMSPropOptimizer,
    "Adagrad": qml.AdagradOptimizer,
}

_OPTIMIZER_ALIASES = {
    "adam": "Adam",
    "gradientdescent": "GradientDescent",
    "gradient_descent": "GradientDescent",
    "gd": "GradientDescent",
    "momentum": "Momentum",
    "nesterov": "NesterovMomentum",
    "nesterovmomentum": "NesterovMomentum",
    "rmsprop": "RMSProp",
    "adagrad": "Adagrad",
}


def _normalize_optimizer_key(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch not in " _-")


def canonicalize_optimizer_name(name: str) -> str:
    """Map case/spacing variants and legacy aliases to canonical registry names."""
    normalized = str(name).strip()
    normalized_key = _normalize_optimizer_key(normalized)

    lookup = {
        _normalize_optimizer_key("Adam"): "Adam",
        _normalize_optimizer_key("GradientDescent"): "GradientDescent",
        _normalize_optimizer_key("Momentum"): "Momentum",
        _normalize_optimizer_key("NesterovMomentum"): "NesterovMomentum",
        _normalize_optimizer_key("RMSProp"): "RMSProp",
        _normalize_optimizer_key("Adagrad"): "Adagrad",
        _normalize_optimizer_key("gd"): "GradientDescent",
        _normalize_optimizer_key("Nesterov"): "NesterovMomentum",
    }

    alias = _OPTIMIZER_ALIASES.get(normalized.lower())
    if alias is not None:
        return alias
    return lookup.get(normalized_key, normalized)


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
    name = canonicalize_optimizer_name(name)
    if name in OPTIMIZERS:
        return OPTIMIZERS[name](stepsize)

    raise ValueError(
        f"Unknown optimizer '{name}'. Available: {', '.join(OPTIMIZERS.keys())}"
    )
