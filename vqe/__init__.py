"""
vqe.__init__.py
---
A modular Variational Quantum Eigensolver (VQE) and SSVQE toolkit built on PennyLane.

Public API:
    - run_vqe
    - run_ssvqe
    - run_vqe_noise_sweep
    - run_vqe_optimizer_comparison
    - run_vqe_ansatz_comparison
    - run_vqe_multi_seed_noise
    - run_vqe_geometry_scan
    - run_vqe_mapping_comparison

    - get_ansatz, init_params, ANSATZES
    - get_optimizer
    - build_hamiltonian, generate_geometry
    - make_run_config_dict, run_signature, save_run_record, ensure_dirs

    - Plotting helpers (visualize.*)
"""

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("vqe-pennylane")

# ------------------------------------------------------------------
# Core VQE APIs
# ------------------------------------------------------------------
from .core import (
    run_vqe,
    run_vqe_noise_sweep,
    run_vqe_optimizer_comparison,
    run_vqe_ansatz_comparison,
    run_vqe_multi_seed_noise,
    run_vqe_geometry_scan,
    run_vqe_mapping_comparison,
)

# ------------------------------------------------------------------
# Ansatz registry & utilities
# ------------------------------------------------------------------
from .ansatz import get_ansatz, init_params, ANSATZES

# ------------------------------------------------------------------
# Optimizers
# ------------------------------------------------------------------
from .optimizer import get_optimizer

# ------------------------------------------------------------------
# Hamiltonian & geometry
# ------------------------------------------------------------------
from .hamiltonian import build_hamiltonian, generate_geometry

# ------------------------------------------------------------------
# I/O utilities (config, hashing, results)
# ------------------------------------------------------------------
from .io_utils import (
    make_run_config_dict,
    run_signature,
    save_run_record,
    ensure_dirs,
)

# ------------------------------------------------------------------
# Visualization utilities
# ------------------------------------------------------------------
from .visualize import (
    plot_convergence,
    plot_ssvqe_convergence_multi,
    plot_optimizer_comparison,
    plot_ansatz_comparison,
    plot_noise_statistics,
)

# ------------------------------------------------------------------
# SSVQE
# ------------------------------------------------------------------
from .ssvqe import run_ssvqe

# ------------------------------------------------------------------
# VQD
# ------------------------------------------------------------------
from .vqd import run_vqd

__all__ = [
    # Core VQE API
    "run_vqe",
    "run_vqe_noise_sweep",
    "run_vqe_optimizer_comparison",
    "run_vqe_ansatz_comparison",
    "run_vqe_multi_seed_noise",
    "run_vqe_geometry_scan",
    "run_vqe_mapping_comparison",
    # Ansatz tools
    "get_ansatz",
    "init_params",
    "ANSATZES",
    # Optimizers
    "get_optimizer",
    # Hamiltonian
    "build_hamiltonian",
    "generate_geometry",
    # I/O
    "make_run_config_dict",
    "run_signature",
    "save_run_record",
    "ensure_dirs",
    # Excited state methods
    "run_ssvqe",
    "run_vqd",
    # Visualization
    "plot_convergence",
    "plot_optimizer_comparison",
    "plot_ansatz_comparison",
    "plot_noise_statistics",
    "plot_ssvqe_convergence_multi",
]
