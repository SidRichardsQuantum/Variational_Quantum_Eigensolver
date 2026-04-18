"""
common.metrics
==============

Shared numerical metrics used across solver packages and notebooks.
"""

from __future__ import annotations

import numpy as np


def compute_fidelity(pure_state, state_or_rho) -> float:
    """
    Fidelity between a pure reference state |psi> and either:
        - a statevector |phi>
        - or a density matrix rho

    Returns |<psi|phi>|^2 or <psi|rho|psi> respectively.
    """
    state_or_rho = np.asarray(state_or_rho, dtype=complex)
    pure_state = np.asarray(pure_state, dtype=complex)
    pure_state = pure_state / np.linalg.norm(pure_state)

    if state_or_rho.ndim == 1:
        state_or_rho = state_or_rho / np.linalg.norm(state_or_rho)
        return float(abs(np.vdot(pure_state, state_or_rho)) ** 2)

    if state_or_rho.ndim == 2:
        trace = np.trace(state_or_rho)
        if abs(trace) > 0.0:
            state_or_rho = state_or_rho / trace
        return float(np.real(np.vdot(pure_state, state_or_rho @ pure_state)))

    raise ValueError("Invalid state shape for fidelity computation")
