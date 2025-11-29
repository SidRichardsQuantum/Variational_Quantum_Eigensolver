"""
vqe.ansatz
----------
Library of parameterized quantum circuits (ansatzes) used in the VQE workflow.

Includes:
- Simple 2-qubit toy ansatzes (RY-CZ, Minimal, TwoQubit-RY-CNOT)
- Hardware-efficient templates (StronglyEntanglingLayers)
- Chemistry-inspired UCCSD / UCCD-style ansatzes
- Parameter initialisation logic aligned with legacy notebooks
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem


# ================================================================
# BASIC ANSATZ CIRCUITS (TOY / HARDWARE-EFFICIENT)
# ================================================================
def two_qubit_ry_cnot(params, wires):
    """
    Two-qubit toy entangler; not a chemically meaningful ansatz.

    Matches the behaviour used in the H₂ noise-scan notebook.
    """
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(-params[0], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def ry_cz(params, wires):
    """
    Single-layer RY rotations followed by a CZ chain.

    Matches the legacy vqe_utils.ry_cz used for H₂ optimizer / ansatz comparisons.
    """
    if len(params) != len(wires):
        raise ValueError(
            f"RY-CZ expects one parameter per wire "
            f"(got {len(params)} vs {len(wires)})"
        )

    for i, w in enumerate(wires):
        qml.RY(params[i], wires=w)

    for i in range(len(wires) - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])


def minimal(params, wires):
    """
    Minimal 2-qubit circuit: RY rotation + CNOT.

    Matches the legacy vqe_utils.minimal used in H₂ ansatz comparisons.
    """
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])


def hardware_efficient_ansatz(params, wires):
    """
    Standard hardware-efficient ansatz using StronglyEntanglingLayers.

    Shape convention:
        params.shape = (n_layers, len(wires), 3)
    """
    qml.templates.StronglyEntanglingLayers(params, wires=wires)


# ================================================================
# UCC-STYLE CHEMISTRY ANSATZES
# ================================================================

def _ucc_cache_key(symbols, coordinates, basis: str):
    coords = np.array(coordinates, dtype=float).flatten().tolist()
    return (tuple(symbols), tuple(coords), basis.upper())


def _build_ucc_data(symbols, coordinates, basis="STO-3G"):
    """
    Compute (singles, doubles, hf_state) for a given molecule and cache them.

    This mirrors what the old notebooks did via:
        - qchem.hf_state(electrons, qubits)
        - qchem.excitations(electrons, qubits)
    """
    if symbols is None or coordinates is None:
        raise ValueError(
            "UCC ansatz requires symbols and coordinates. "
            "Make sure build_hamiltonian(...) is used and passed through."
        )

    key = _ucc_cache_key(symbols, coordinates, basis)

    if not hasattr(_build_ucc_data, "_cache"):
        _build_ucc_data._cache = {}

    if key not in _build_ucc_data._cache:
        try:
            mol = qchem.Molecule(symbols, coordinates, charge=0, basis=basis)
        except TypeError:
            # Older PennyLane versions without basis kwarg
            mol = qchem.Molecule(symbols, coordinates, charge=0)

        electrons = mol.n_electrons
        spin_orbitals = 2 * mol.n_orbitals

        singles, doubles = qchem.excitations(electrons, spin_orbitals)
        hf_state = qchem.hf_state(electrons, spin_orbitals)

        _build_ucc_data._cache[key] = (singles, doubles, hf_state)

    return _build_ucc_data._cache[key]


def uccsd_ansatz(params, wires, symbols=None, coordinates=None, basis="STO-3G"):
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    Behaviour is chosen to match legacy notebooks where we used:

        excitation_ansatz(
            params,
            wires=range(qubits),
            hf_state=hf,
            excitations=(singles, doubles),
            excitation_type="both",
        )

    Args:
        params: 1D array of length len(singles) + len(doubles)
        wires: sequence of PennyLane wires
        symbols, coordinates, basis: molecular data (required for chemistry use)
    """
    singles, doubles, hf_state = _build_ucc_data(symbols, coordinates, basis=basis)

    num_wires = len(wires)
    if len(hf_state) != num_wires:
        raise ValueError(
            f"HF state length ({len(hf_state)}) does not match number of wires ({num_wires})."
        )

    qml.BasisState(np.array(hf_state, dtype=int), wires=wires)

    n_singles = len(singles)
    n_doubles = len(doubles)
    if len(params) != n_singles + n_doubles:
        raise ValueError(
            f"UCCSD expects {n_singles + n_doubles} parameters "
            f"(got {len(params)})."
        )

    # Singles
    for i, exc in enumerate(singles):
        qml.SingleExcitation(params[i], wires=list(exc))

    # Doubles
    for j, exc in enumerate(doubles):
        qml.DoubleExcitation(params[n_singles + j], wires=list(exc))


def uccd_ansatz(params, wires, symbols=None, coordinates=None, basis="STO-3G"):
    """
    UCC Doubles-only ansatz (UCCD / "UCC-D").

    Designed to mirror the LiH notebook behaviour where we used
    `excitation_ansatz(..., excitation_type="double")` with zero initial params.

    Args:
        params: 1D array of length len(doubles)
        wires: sequence of wires
        symbols, coordinates, basis: molecular data
    """
    singles, doubles, hf_state = _build_ucc_data(symbols, coordinates, basis=basis)

    num_wires = len(wires)
    if len(hf_state) != num_wires:
        raise ValueError(
            f"HF state length ({len(hf_state)}) does not match number of wires ({num_wires})."
        )

    qml.BasisState(np.array(hf_state, dtype=int), wires=wires)

    if len(params) != len(doubles):
        raise ValueError(
            f"UCCD expects {len(doubles)} parameters (got {len(params)})."
        )

    for i, exc in enumerate(doubles):
        qml.DoubleExcitation(params[i], wires=list(exc))


# ================================================================
# REGISTRY
# ================================================================

ANSATZES = {
    "TwoQubit-RY-CNOT": two_qubit_ry_cnot,
    "RY-CZ": ry_cz,
    "Minimal": minimal,
    "StronglyEntanglingLayers": hardware_efficient_ansatz,
    "UCCSD": uccsd_ansatz,
    "UCC-D": uccd_ansatz,
    "UCCD": uccd_ansatz,  # alias
}


def get_ansatz(name: str):
    """Return ansatz function by name."""
    if name not in ANSATZES:
        available = ", ".join(sorted(ANSATZES.keys()))
        raise ValueError(f"Unknown ansatz '{name}'. Available: {available}")
    return ANSATZES[name]


# ================================================================
# PARAMETER INITIALISATION
# ================================================================
def init_params(
    ansatz_name: str,
    num_wires: int,
    scale: float = 0.01,
    requires_grad: bool = True,
    symbols=None,
    coordinates=None,
    basis: str = "STO-3G",
    seed: int = 0,
):
    """
    Initialise variational parameters for a given ansatz.

    Design choices (to match legacy notebooks):

    - TwoQubit-RY-CNOT / Minimal:
        * 1 parameter, small random normal ~ N(0, scale²)
    - RY-CZ:
        * `num_wires` parameters, random normal ~ N(0, scale²)
    - StronglyEntanglingLayers:
        * params.shape = (1, num_wires, 3), normal with width ~ π
    - UCCSD / UCCD / UCC-D:
        * **All zeros**, matching old notebooks where VQE started from θ = 0
          and optimised UCCSD/UCCD from that point.

    Returns:
        np.ndarray with `requires_grad=True`.
    """
    np.random.seed(seed)

    # --- Toy ansatzes ---
    if ansatz_name in ["TwoQubit-RY-CNOT", "Minimal"]:
        vals = scale * np.random.randn(1)

    elif ansatz_name == "RY-CZ":
        vals = scale * np.random.randn(num_wires)

    elif ansatz_name == "StronglyEntanglingLayers":
        # One layer by default; can be generalised later if needed
        vals = np.random.normal(0.0, np.pi, (1, num_wires, 3))

    # --- Chemistry ansatzes (UCC family) ---
    elif ansatz_name in ["UCCSD", "UCC-D", "UCCD"]:
        # We need molecular data to know #excitations.
        # For pure "H2", "LiH", etc., these are supplied by build_hamiltonian + engine.build_ansatz.
        if symbols is None or coordinates is None:
            raise ValueError(
                f"Ansatz '{ansatz_name}' requires symbols/coordinates "
                "to determine excitation count. Ensure you are using "
                "build_hamiltonian(...) and engine.build_ansatz(...)."
            )

        # Compute excitations exactly as in the notebooks
        try:
            mol = qchem.Molecule(symbols, coordinates, charge=0, basis=basis)
        except TypeError:
            mol = qchem.Molecule(symbols, coordinates, charge=0)

        electrons = mol.n_electrons
        spin_orbitals = 2 * mol.n_orbitals
        singles, doubles = qchem.excitations(electrons, spin_orbitals)

        if ansatz_name in ["UCC-D", "UCCD"]:
            # Doubles-only: match LiH notebook – all zeros
            vals = np.zeros(len(doubles))
        else:
            # UCCSD: singles + doubles, all zeros (legacy behaviour)
            vals = np.zeros(len(singles) + len(doubles))

    else:
        available = ", ".join(sorted(ANSATZES.keys()))
        raise ValueError(f"Unknown ansatz '{ansatz_name}'. Available: {available}")

    return np.array(vals, requires_grad=requires_grad)
