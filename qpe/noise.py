"""
qpe/noise.py
============
Noise utility functions for Quantum Phase Estimation (QPE) circuits.

Currently supports:
  • Depolarizing channel
  • Amplitude damping channel

Both can be applied together to all wires in a circuit segment.
"""

from __future__ import annotations

from typing import Iterable

import pennylane as qml


def apply_noise_all(
    wires: Iterable[int],
    p_dep: float = 0.0,
    p_amp: float = 0.0,
) -> None:
    """Apply depolarizing and/or amplitude damping noise to the given wires.

    This function is intended to be called *inside* a QNode, typically
    after a unitary operation or ansatz to simulate mixed noise channels.

    Parameters
    ----------
    wires : Iterable[int]
        Wires (qubit indices) on which to apply noise.
    p_dep : float, optional
        Depolarizing probability per wire (default = 0.0).
    p_amp : float, optional
        Amplitude damping probability per wire (default = 0.0).

    Example
    -------
    >>> apply_noise_all([0, 1, 2], p_dep=0.02, p_amp=0.01)
    """
    if p_dep <= 0.0 and p_amp <= 0.0:
        return

    for w in wires:
        if p_dep > 0.0:
            qml.DepolarizingChannel(p_dep, wires=w)
        if p_amp > 0.0:
            qml.AmplitudeDamping(p_amp, wires=w)
