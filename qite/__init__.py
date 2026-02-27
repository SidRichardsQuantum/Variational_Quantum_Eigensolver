"""
qite.__init__.py
----------------
Public API surface for the QITE / VarQITE subpackage.

This package provides:
- Variational imaginary-time evolution (VarQITE) workflows
- Reproducible caching and I/O helpers
- Plotting utilities for QITE notebooks
- Circuit / device plumbing for advanced users

Design notes
------------
- VarQITE parameter updates are pure-state only (McLachlan principle).
- Noise is supported only for post-evaluation, not for parameter updates.
- The public surface mirrors vqe/qpe where appropriate, but remains minimal.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

# ---------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------
try:
    __version__ = _pkg_version("vqe-pennylane")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


# ---------------------------------------------------------------------
# Public API Imports
# ---------------------------------------------------------------------
from .core import run_qite  # noqa: E402

from .engine import (  # noqa: E402
    build_ansatz,
    make_device,
    make_energy_qnode,
    make_state_qnode,
    qite_step,
)

from .hamiltonian import build_hamiltonian  # noqa: E402

from .io_utils import (  # noqa: E402
    ensure_dirs,
    make_filename_prefix,
    make_run_config_dict,
    run_signature,
    save_run_record,
)

from .visualize import (  # noqa: E402
    plot_convergence,
    plot_diagnostics,
    plot_noise_statistics,
)

# ---------------------------------------------------------------------
# Public API Surface
# ---------------------------------------------------------------------
__all__ = [
    # Package metadata
    "__version__",
    # Core workflow
    "run_qite",
    # Hamiltonian
    "build_hamiltonian",
    # Engine utilities (advanced / notebooks)
    "make_device",
    "make_energy_qnode",
    "make_state_qnode",
    "build_ansatz",
    "qite_step",
    # I/O helpers
    "ensure_dirs",
    "make_run_config_dict",
    "run_signature",
    "save_run_record",
    "make_filename_prefix",
    # Plotting
    "plot_convergence",
    "plot_noise_statistics",
    "plot_diagnostics",
]
