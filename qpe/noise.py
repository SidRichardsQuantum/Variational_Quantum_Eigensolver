"""
qpe/noise.py
============
Noise utility functions for Quantum Phase Estimation (QPE) circuits.

Currently supports:
  • Depolarizing channel
  • Amplitude damping channel
  • Phase damping channel
  • Bit-flip channel
  • Phase-flip channel
"""

from __future__ import annotations

from typing import Iterable

from common.noise import apply_builtin_noise


def apply_noise_all(
    wires: Iterable[int],
    p_dep: float = 0.0,
    p_amp: float = 0.0,
    p_phase_damp: float = 0.0,
    p_bit_flip: float = 0.0,
    p_phase_flip: float = 0.0,
) -> None:
    """Apply built-in single-qubit noise channels to the given wires.

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
    apply_builtin_noise(
        wires,
        depolarizing_prob=float(p_dep),
        amplitude_damping_prob=float(p_amp),
        phase_damping_prob=float(p_phase_damp),
        bit_flip_prob=float(p_bit_flip),
        phase_flip_prob=float(p_phase_flip),
    )
