"""
VQE Module

This module provides functionalities to run Variational Quantum Eigensolver (VQE) simulations
using PennyLane. It includes core routines, ansatz definitions, optimizers, Hamiltonian
construction, I/O utilities, and visualization tools.

Example:
    >>> from vqe import run_vqe, plot_convergence
    >>> result = run_vqe(hamiltonian, ansatz, optimizer, init_params)
    >>> plot_convergence(result)
"""

__version__ = "0.1.0"

from .core import (
    run_vqe,
    run_vqe_noise_sweep,
    run_vqe_optimizer_comparison,
    run_vqe_ansatz_comparison,
    run_vqe_multi_seed_noise,
    run_vqe_geometry_scan,
    run_vqe_mapping_comparison,
)
from .ansatz import get_ansatz, init_params, ANSATZES
from .optimizer import minimize_energy, get_optimizer
from .hamiltonian import build_hamiltonian, generate_geometry
from .io_utils import (
    make_run_config_dict,
    run_signature,
    save_run_record,
    ensure_dirs,
)
from .visualize import (
    plot_convergence,
    plot_noise_sweep,
    plot_optimizer_comparison,
    plot_ansatz_comparison,
    plot_noise_statistics,
)

__all__ = [
    "run_vqe",
    "run_vqe_noise_sweep",
    "run_vqe_optimizer_comparison",
    "run_vqe_ansatz_comparison",
    "run_vqe_multi_seed_noise",
    "run_vqe_geometry_scan",
    "run_vqe_mapping_comparison",
    "get_ansatz",
    "init_params",
    "ANSATZES",
    "minimize_energy",
    "get_optimizer",
    "build_hamiltonian",
    "generate_geometry",
    "make_run_config_dict",
    "run_signature",
    "save_run_record",
    "ensure_dirs",
    "plot_convergence",
    "plot_noise_sweep",
    "plot_optimizer_comparison",
    "plot_ansatz_comparison",
    "plot_noise_statistics",
]

from .ssvqe import run_ssvqe
__all__.append("run_ssvqe")

from .engine import make_device, build_ansatz, build_optimizer
from .ssvqe import run_ssvqe

__all__ += [
    "make_device", "build_ansatz", "build_optimizer",
    "run_ssvqe",
]
