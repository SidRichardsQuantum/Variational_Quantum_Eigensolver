"""
qpe.__init__.py
===
Quantum Phase Estimation (QPE) module of the VQE/QPE PennyLane simulation suite.

Primary user-facing API:
    - run_qpe()
    - plot_qpe_distribution()
    - plot_qpe_sweep()
    - save_qpe_result()
    - load_qpe_result()
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

from common import mpl_env as _mpl_env  # noqa: F401
from common.lazy import LazyExports, list_exports, load_export

try:
    __version__ = _pkg_version("vqe-pennylane")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__docformat__ = "restructuredtext"

_EXPORTS: LazyExports = {
    # Hamiltonian
    "build_hamiltonian": ("qpe.hamiltonian", "build_hamiltonian"),
    # Core QPE
    "run_qpe": ("qpe.core", "run_qpe"),
    "bitstring_to_phase": ("qpe.core", "bitstring_to_phase"),
    "phase_to_energy_unwrapped": ("qpe.core", "phase_to_energy_unwrapped"),
    "hartree_fock_energy": ("qpe.core", "hartree_fock_energy"),
    # Visualization
    "plot_qpe_distribution": ("qpe.visualize", "plot_qpe_distribution"),
    "plot_qpe_sweep": ("qpe.visualize", "plot_qpe_sweep"),
    # I/O + Caching
    "save_qpe_result": ("qpe.io_utils", "save_qpe_result"),
    "load_qpe_result": ("qpe.io_utils", "load_qpe_result"),
    "signature_hash": ("qpe.io_utils", "signature_hash"),
    # Noise
    "apply_noise_all": ("qpe.noise", "apply_noise_all"),
}

__all__ = [
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
