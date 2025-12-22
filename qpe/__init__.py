"""
qpe
===
Quantum Phase Estimation (QPE) module of the VQE/QPE PennyLane simulation suite.

This subpackage provides:
    • Unified Hamiltonian construction for molecules
    • Noiseless and noisy Quantum Phase Estimation (QPE)
    • Probability distribution & sweep plotting (unified with VQE)
    • JSON-based caching and reproducible run signatures
    • Noise channels and controlled time evolution utilities

Primary user-facing API:
    - run_qpe()
    - plot_qpe_distribution()
    - plot_qpe_sweep()
    - save_qpe_result()
    - load_qpe_result()
"""

from importlib.metadata import version as _pkg_version

__version__ = _pkg_version("vqe-pennylane")
__docformat__ = "restructuredtext"


# ---------------------------------------------------------------------
# Public API Imports
# ---------------------------------------------------------------------
from .hamiltonian import (  # noqa: F401
    build_hamiltonian,
)

from .core import (  # noqa: F401
    run_qpe,
    bitstring_to_phase,
    phase_to_energy_unwrapped,
    hartree_fock_energy,
)

from .visualize import (  # noqa: F401
    plot_qpe_distribution,
    plot_qpe_sweep,
)

from .io_utils import (  # noqa: F401
    save_qpe_result,
    load_qpe_result,
    signature_hash,
)

from .noise import apply_noise_all  # noqa: F401


# ---------------------------------------------------------------------
# Public API Surface
# ---------------------------------------------------------------------
__all__ = [
    # Hamiltonian
    "build_hamiltonian",
    # Core QPE
    "run_qpe",
    "bitstring_to_phase",
    "phase_to_energy_unwrapped",
    "hartree_fock_energy",
    # Visualization
    "plot_qpe_distribution",
    "plot_qpe_sweep",
    # I/O + Caching
    "save_qpe_result",
    "load_qpe_result",
    "signature_hash",
    # Noise
    "apply_noise_all",
]
