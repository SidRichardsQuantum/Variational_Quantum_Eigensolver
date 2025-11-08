"""
Noise utilities for Quantum Phase Estimation (QPE) simulations.

Currently supports:
    • Depolarizing channel
    • Amplitude damping channel

These can be combined for mixed-noise simulations.
"""

import pennylane as qml


def apply_noise_all(wires, p_dep: float = 0.0, p_amp: float = 0.0):
    """
    Apply depolarizing and amplitude damping noise to all specified wires.

    Args:
        wires (Iterable[int]): List of wires (qubit indices) to which noise is applied.
        p_dep (float): Depolarizing channel probability per wire (default=0.0)
        p_amp (float): Amplitude damping probability per wire (default=0.0)

    Example:
        >>> apply_noise_all([0, 1, 2], p_dep=0.02, p_amp=0.01)
    """
    for w in wires:
        if p_dep > 0.0:
            qml.DepolarizingChannel(p_dep, wires=w)
        if p_amp > 0.0:
            qml.AmplitudeDamping(p_amp, wires=w)
