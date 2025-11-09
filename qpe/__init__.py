"""
QPE module for the Variational Quantum Eigensolver project.

This package provides core functionality, visualization, noise modeling, and
file utilities for running and analyzing QPE simulations using PennyLane.

Example:
    >>> from qpe import run_qpe, plot_qpe_distribution
    >>> result = run_qpe(hamiltonian, hf_state, n_ancilla=4)
    >>> plot_qpe_distribution(result)
"""

__version__ = "0.1.0"

from .core import (
    run_qpe,
    hartree_fock_energy,
    bitstring_to_phase,
    phase_to_energy_unwrapped,
)
from .visualize import (
    plot_qpe_distribution,
    plot_qpe_sweep,
    save_qpe_plot,
)
from .io_utils import (
    save_qpe_result,
    load_qpe_result,
    signature_hash,
    ensure_dirs,
)
from .noise import apply_noise_all

__all__ = [
    # Core
    "run_qpe",
    "hartree_fock_energy",
    "bitstring_to_phase",
    "phase_to_energy_unwrapped",
    # Visualization
    "plot_qpe_distribution",
    "plot_qpe_sweep",
    "save_qpe_plot",
    # I/O
    "save_qpe_result",
    "load_qpe_result",
    "signature_hash",
    "ensure_dirs",
    # Noise
    "apply_noise_all",
]
