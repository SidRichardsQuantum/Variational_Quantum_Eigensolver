"""
common.__init__.py
======

Shared utilities used across VQE, QPE, and future solvers.

The public exports are resolved lazily so importing high-level packages for
metadata or CLI help does not also import the quantum chemistry stack.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any

from . import mpl_env as _mpl_env  # noqa: F401
from .lazy import LazyExports, list_exports, load_export

try:
    __version__ = _pkg_version("vqe-pennylane")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


_EXPORTS: LazyExports = {
    # Molecules
    "MOLECULES": ("common.molecules", "MOLECULES"),
    "get_molecule_config": ("common.molecules", "get_molecule_config"),
    # Geometry
    "generate_geometry": ("common.geometry", "generate_geometry"),
    # Hamiltonian
    "build_hamiltonian": ("common.hamiltonian", "build_hamiltonian"),
    "get_exact_spectrum": ("common.hamiltonian", "get_exact_spectrum"),
    "summarize_registry_coverage": (
        "common.hamiltonian",
        "summarize_registry_coverage",
    ),
    # Benchmarks/helpers
    "timed_call": ("common.benchmarks", "timed_call"),
    "summary_stats": ("common.benchmarks", "summary_stats"),
    "qpe_branch_candidates": ("common.benchmarks", "qpe_branch_candidates"),
    "analyze_qpe_result": ("common.benchmarks", "analyze_qpe_result"),
    "qpe_calibration_plan": ("common.benchmarks", "qpe_calibration_plan"),
    "exact_ground_energy_for_problem": (
        "common.benchmarks",
        "exact_ground_energy_for_problem",
    ),
    "ionization_energy_panel": ("common.benchmarks", "ionization_energy_panel"),
    "summarize_problem": ("common.benchmarks", "summarize_problem"),
    "compare_benchmark_runs": ("common.benchmarks", "compare_benchmark_runs"),
    "list_benchmark_suites": ("common.benchmarks", "list_benchmark_suites"),
    "run_benchmark_suite": ("common.benchmarks", "run_benchmark_suite"),
    "environment_metadata": ("common.environment", "environment_metadata"),
    "ensure_environment_metadata": (
        "common.environment",
        "ensure_environment_metadata",
    ),
    "compute_fidelity": ("common.metrics", "compute_fidelity"),
    "ResolvedProblem": ("common.problem", "ResolvedProblem"),
    "resolve_problem": ("common.problem", "resolve_problem"),
    # Plotting
    "build_filename": ("common.plotting", "build_filename"),
    "save_plot": ("common.plotting", "save_plot"),
    "format_molecule_name": ("common.naming", "format_molecule_name"),
    "format_token": ("common.naming", "format_token"),
    # Molecule visualization
    "plot_molecule": ("common.molecule_viz", "plot_molecule"),
    "infer_bonds": ("common.molecule_viz", "infer_bonds"),
    "infer_angles_from_bonds": ("common.molecule_viz", "infer_angles_from_bonds"),
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
