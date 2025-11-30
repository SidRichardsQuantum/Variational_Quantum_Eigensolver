from __future__ import annotations

__version__ = "0.2.0"

"""
vqe_qpe_common
======

Shared utilities used across VQE, QPE, and future solvers.

This subpackage contains:
    • vqe_qpe_common.molecules   — canonical molecule registry
    • vqe_qpe_common.geometry    — unified geometry generators (bond/angle scans)
    • vqe_qpe_common.hamiltonian — single source of truth for Hamiltonian construction
    • vqe_qpe_common.plotting    — global plotting + filename/dir management

All high-level solvers (VQE, QPE, QSVT, etc.) should import molecule
definitions, geometry logic, Hamiltonians, and plotting helpers from here
to avoid duplication and ensure reproducibility.

Example
-------
    from vqe_qpe_common import (
        MOLECULES,
        get_molecule_config,
        generate_geometry,
        build_hamiltonian,
        build_filename,
        save_plot,
    )
"""

# Molecule data + helpers
from .molecules import MOLECULES, get_molecule_config  # noqa: F401

# Geometry (bond length, angle scans, parametrized coordinates)
from .geometry import generate_geometry  # noqa: F401

# Hamiltonian construction (PennyLane + OpenFermion fallback)
from .hamiltonian import build_hamiltonian  # noqa: F401

# Plotting utilities shared across VQE + QPE
from .plotting import (
    build_filename,
    save_plot,
    format_molecule_name,
    format_token,
)  # noqa: F401


__all__ = [
    # Molecules
    "MOLECULES",
    "get_molecule_config",

    # Geometry
    "generate_geometry",

    # Hamiltonian
    "build_hamiltonian",

    # Plotting
    "build_filename",
    "save_plot",
    "format_molecule_name",
    "format_token",
]
