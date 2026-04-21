"""
qite.__init__.py
----------------
Public API surface for the QITE / projected-dynamics subpackage.

Public entrypoints are resolved lazily so command-line help remains lightweight
while preserving ``from qite import run_qite`` style imports.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

from common import mpl_env as _mpl_env  # noqa: F401
from common.lazy import LazyExports, list_exports, load_export

# ---------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------
try:
    __version__ = _pkg_version("vqe-pennylane")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


_EXPORTS: LazyExports = {
    # Core workflow
    "run_qite": ("qite.core", "run_qite"),
    "run_qrte": ("qite.core", "run_qrte"),
    # Hamiltonian
    "build_hamiltonian": ("qite.hamiltonian", "build_hamiltonian"),
    # Engine utilities (advanced / notebooks)
    "make_device": ("qite.engine", "make_device"),
    "make_energy_qnode": ("qite.engine", "make_energy_qnode"),
    "make_state_qnode": ("qite.engine", "make_state_qnode"),
    "build_ansatz": ("qite.engine", "build_ansatz"),
    "qite_step": ("qite.engine", "qite_step"),
    "qrte_step": ("qite.engine", "qrte_step"),
    # I/O helpers
    "ensure_dirs": ("qite.io_utils", "ensure_dirs"),
    "make_run_config_dict": ("qite.io_utils", "make_run_config_dict"),
    "run_signature": ("qite.io_utils", "run_signature"),
    "save_run_record": ("qite.io_utils", "save_run_record"),
    "make_filename_prefix": ("qite.io_utils", "make_filename_prefix"),
    # Plotting
    "plot_convergence": ("qite.visualize", "plot_convergence"),
    "plot_noise_statistics": ("qite.visualize", "plot_noise_statistics"),
    "plot_diagnostics": ("qite.visualize", "plot_diagnostics"),
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
