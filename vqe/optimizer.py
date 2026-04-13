"""
vqe.optimizer
-------------
Lightweight wrapper over PennyLane optimizers with a unified interface.

Provides:
    - get_optimizer(name, stepsize)
    - get_optimizer_stepsize(name)
"""

from __future__ import annotations

from typing import Any

import pennylane as qml

# ================================================================
# AVAILABLE OPTIMIZERS
# ================================================================
OPTIMIZERS: dict[str, dict[str, Any]] = {
    "Adam": {
        "factory": qml.AdamOptimizer,
        "stepsize": 0.15,
        "aliases": ("adam",),
    },
    "GradientDescent": {
        "factory": qml.GradientDescentOptimizer,
        "stepsize": 0.10,
        "aliases": ("gradientdescent", "gradient_descent", "gd"),
    },
    "Momentum": {
        "factory": qml.MomentumOptimizer,
        "stepsize": 0.10,
        "aliases": ("momentum",),
    },
    "NesterovMomentum": {
        "factory": qml.NesterovMomentumOptimizer,
        "stepsize": 0.20,
        "aliases": ("nesterov", "nesterovmomentum"),
    },
    "RMSProp": {
        "factory": qml.RMSPropOptimizer,
        "stepsize": 0.01,
        "aliases": ("rmsprop",),
    },
    "Adagrad": {
        "factory": qml.AdagradOptimizer,
        "stepsize": 0.10,
        "aliases": ("adagrad",),
    },
}

_OPTIMIZER_ALIASES = {
    alias: canonical
    for canonical, spec in OPTIMIZERS.items()
    for alias in spec["aliases"]
}


def _normalize_optimizer_key(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch not in " _-")


def canonicalize_optimizer_name(name: str) -> str:
    """Map case/spacing variants and legacy aliases to canonical registry names."""
    normalized = str(name).strip()
    normalized_key = _normalize_optimizer_key(normalized)

    lookup = {
        _normalize_optimizer_key(canonical): canonical for canonical in OPTIMIZERS
    }

    alias = _OPTIMIZER_ALIASES.get(normalized.lower())
    if alias is not None:
        return alias
    return lookup.get(normalized_key, normalized)


def get_optimizer_stepsize(name: str = "Adam") -> float:
    """
    Return the calibrated default stepsize for an optimizer.

    Args:
        name: Optimizer identifier (case-insensitive).

    Returns:
        Calibrated default learning rate for the canonical optimizer.
    """
    canonical_name = canonicalize_optimizer_name(name)
    if canonical_name in OPTIMIZERS:
        return float(OPTIMIZERS[canonical_name]["stepsize"])

    raise ValueError(
        f"Unknown optimizer '{canonical_name}'. Available: {', '.join(OPTIMIZERS.keys())}"
    )


# ================================================================
# MAIN FACTORY
# ================================================================
def get_optimizer(name: str = "Adam", stepsize: float | None = None):
    """
    Return a PennyLane optimizer instance by name.

    Args:
        name: Optimizer identifier (case-insensitive).
        stepsize: Learning rate. If omitted, use the calibrated optimizer default.

    Returns:
        An instantiated optimizer.
    """
    canonical_name = canonicalize_optimizer_name(name)
    if canonical_name in OPTIMIZERS:
        resolved_stepsize = (
            get_optimizer_stepsize(canonical_name)
            if stepsize is None
            else float(stepsize)
        )
        return OPTIMIZERS[canonical_name]["factory"](resolved_stepsize)

    raise ValueError(
        f"Unknown optimizer '{canonical_name}'. Available: {', '.join(OPTIMIZERS.keys())}"
    )
