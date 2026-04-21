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
- Public entrypoints are resolved lazily so command-line help does not import
  PennyLane/OpenFermion-heavy solver modules.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

from common import mpl_env as _mpl_env  # noqa: F401
from common.lazy import LazyExports, list_exports, load_export

# Version ----------------------------------------------------------------------
try:
    __version__ = _pkg_version("vqe-pennylane")
except PackageNotFoundError:  # pragma: no cover
    # Allows editable installs / local runs without installed dist metadata.
    __version__ = "0.0.0"


_EXPORTS: LazyExports = {
    # Core VQE workflows
    "run_vqe": ("vqe.core", "run_vqe"),
    "run_vqe_optimizer_comparison": ("vqe.core", "run_vqe_optimizer_comparison"),
    "run_vqe_ansatz_comparison": ("vqe.core", "run_vqe_ansatz_comparison"),
    "run_vqe_multi_seed_noise": ("vqe.core", "run_vqe_multi_seed_noise"),
    "run_vqe_geometry_scan": ("vqe.core", "run_vqe_geometry_scan"),
    "run_vqe_low_qubit_benchmark": ("vqe.core", "run_vqe_low_qubit_benchmark"),
    "run_vqe_mapping_comparison": ("vqe.core", "run_vqe_mapping_comparison"),
    # Excited-state methods
    "run_ssvqe": ("vqe.ssvqe", "run_ssvqe"),
    "run_vqd": ("vqe.vqd", "run_vqd"),
    "run_lr_vqe": ("vqe.lr_vqe", "run_lr_vqe"),
    "run_qse": ("vqe.qse", "run_qse"),
    "run_eom_vqe": ("vqe.eom_vqe", "run_eom_vqe"),
    "run_eom_qse": ("vqe.eom_qse", "run_eom_qse"),
    # Ansatz / optimizer registries
    "ANSATZES": ("vqe.ansatz", "ANSATZES"),
    "get_ansatz": ("vqe.ansatz", "get_ansatz"),
    "init_params": ("vqe.ansatz", "init_params"),
    "get_optimizer": ("vqe.optimizer", "get_optimizer"),
    "get_optimizer_stepsize": ("vqe.optimizer", "get_optimizer_stepsize"),
    "OPTIMIZERS": ("vqe.optimizer", "OPTIMIZERS"),
    # Hamiltonian / geometry
    "build_hamiltonian": ("vqe.hamiltonian", "build_hamiltonian"),
    "generate_geometry": ("vqe.hamiltonian", "generate_geometry"),
    # I/O helpers
    "make_run_config_dict": ("vqe.io_utils", "make_run_config_dict"),
    "run_signature": ("vqe.io_utils", "run_signature"),
    "save_run_record": ("vqe.io_utils", "save_run_record"),
    "ensure_dirs": ("vqe.io_utils", "ensure_dirs"),
    # Plotting
    "plot_convergence": ("vqe.visualize", "plot_convergence"),
    "plot_optimizer_comparison": ("vqe.visualize", "plot_optimizer_comparison"),
    "plot_ansatz_comparison": ("vqe.visualize", "plot_ansatz_comparison"),
    "plot_noise_statistics": ("vqe.visualize", "plot_noise_statistics"),
    "plot_multi_state_convergence": ("vqe.visualize", "plot_multi_state_convergence"),
    # ADAPT-VQE
    "run_adapt_vqe": ("vqe.adapt", "run_adapt_vqe"),
}

__all__ = [
    # Package metadata
    "__version__",
    *_EXPORTS.keys(),
]


def __getattr__(name: str) -> Any:
    return load_export(
        package_name=__name__,
        package_globals=globals(),
        exports=_EXPORTS,
        name=name,
    )


def __dir__() -> list[str]:
    import sys

    return list_exports(sys.modules[__name__], _EXPORTS)
