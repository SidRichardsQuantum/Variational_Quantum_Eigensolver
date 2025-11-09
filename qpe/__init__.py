"""
qpe
===
Quantum Phase Estimation (QPE) module for the Variational Quantum Eigensolver project.

This package provides:
  • Core QPE simulation (`run_qpe`)
  • Noise modeling (`apply_noise_all`)
  • Visualization tools (`plot_qpe_distribution`, `plot_qpe_sweep`)
  • Caching and result management utilities

Example
-------
    >>> from qpe import run_qpe, plot_qpe_distribution
    >>> from qpe.io_utils import save_qpe_result
    >>> result = run_qpe(hamiltonian, hf_state, n_ancilla=4)
    >>> plot_qpe_distribution(result)
    >>> save_qpe_result(result)

Requirements
------------
PennyLane ≥ 0.34 and PennyLane-qchem must be installed.
Optional: `openfermion` and `openfermionpyscf` for open-shell systems.
"""

__version__ = "0.1.0"
__docformat__ = "restructuredtext"

from .core import (  # noqa: F401
    run_qpe,
    hartree_fock_energy,
    bitstring_to_phase,
    phase_to_energy_unwrapped,
)
from .visualize import (  # noqa: F401
    plot_qpe_distribution,
    plot_qpe_sweep,
    save_qpe_plot,
)
from .io_utils import (  # noqa: F401
    save_qpe_result,
    load_qpe_result,
    signature_hash,
    ensure_dirs,
)
from .noise import apply_noise_all  # noqa: F401


__all__ = [
    # Core
    "run_qpe",
    "hartree_fock_energy",
    "bitstring_to_phase",
    "phase_to_energy_unwrapped",
    # Visualization
    "plot_qpe_distribution",
    "plot_qpe_sweep",
    "save_qpe_plot",
    # I/O
    "save_qpe_result",
    "load_qpe_result",
    "signature_hash",
    "ensure_dirs",
    # Noise
    "apply_noise_all",
]
