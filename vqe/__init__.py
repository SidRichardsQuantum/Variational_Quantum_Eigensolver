"""
vqe.__init__.py
---------------
Public API surface for the VQE subpackage.

This package provides:
- Ground-state VQE workflows (run, sweeps, comparisons, scans)
- Excited-state solvers:
    * SSVQE (joint, shared-parameter subspace method)
    * VQD   (sequential deflation method)
- Shared plotting helpers used across notebooks and CLI

Design notes
------------
- Keep imports lightweight and stable for downstream users.
- Prefer re-exporting the primary entrypoints rather than internal helpers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# Version ----------------------------------------------------------------------
try:
    __version__ = _pkg_version("vqe-pennylane")
except PackageNotFoundError:  # pragma: no cover
    # Allows editable installs / local runs without installed dist metadata.
    __version__ = "0.0.0"


# Core VQE APIs ----------------------------------------------------------------
from .core import (  # noqa: E402
    run_vqe,
    run_vqe_ansatz_comparison,
    run_vqe_geometry_scan,
    run_vqe_mapping_comparison,
    run_vqe_multi_seed_noise,
    run_vqe_noise_sweep,
    run_vqe_optimizer_comparison,
)

# Ansatz registry & utilities ---------------------------------------------------
from .ansatz import ANSATZES, get_ansatz, init_params  # noqa: E402

# Optimizers -------------------------------------------------------------------
from .optimizer import get_optimizer  # noqa: E402

# Hamiltonian & geometry --------------------------------------------------------
from .hamiltonian import build_hamiltonian, generate_geometry  # noqa: E402

# I/O utilities (config, hashing, results) -------------------------------------
from .io_utils import (
    ensure_dirs,
    make_run_config_dict,
    run_signature,
    save_run_record,
)  # noqa: E402

# Visualization utilities -------------------------------------------------------
from .visualize import (  # noqa: E402
    plot_ansatz_comparison,
    plot_convergence,
    plot_multi_state_convergence,
    plot_noise_statistics,
    plot_optimizer_comparison,
)

# Excited-state methods ---------------------------------------------------------
from .ssvqe import run_ssvqe  # noqa: E402
from .vqd import run_vqd  # noqa: E402


__all__ = [
    # Package metadata
    "__version__",
    # Core VQE workflows
    "run_vqe",
    "run_vqe_noise_sweep",
    "run_vqe_optimizer_comparison",
    "run_vqe_ansatz_comparison",
    "run_vqe_multi_seed_noise",
    "run_vqe_geometry_scan",
    "run_vqe_mapping_comparison",
    # Excited-state methods
    "run_ssvqe",
    "run_vqd",
    # Ansatz / optimizer registries
    "ANSATZES",
    "get_ansatz",
    "init_params",
    "get_optimizer",
    # Hamiltonian / geometry
    "build_hamiltonian",
    "generate_geometry",
    # I/O helpers
    "make_run_config_dict",
    "run_signature",
    "save_run_record",
    "ensure_dirs",
    # Plotting
    "plot_convergence",
    "plot_optimizer_comparison",
    "plot_ansatz_comparison",
    "plot_noise_statistics",
    "plot_multi_state_convergence",
]
