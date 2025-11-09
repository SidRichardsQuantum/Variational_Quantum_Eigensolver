"""
vqe.ansatz
----------
Library of parameterized quantum circuits (ansatzes) used in the VQE workflow.

Includes:
- Simple 2-qubit toy ansatzes (RY-CZ, Minimal)
- Hardware-efficient templates
- Chemistry-inspired UCCSD and UCCD ansatzes (cached for speed)
- Parameter initialization logic
"""

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem


# ================================================================
# BASIC ANSATZ CIRCUITS
# ================================================================
def two_qubit_ry_cnot(params, wires):
    """Two-qubit toy entangler; not a chemically meaningful ansatz."""
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(-params[0], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def ry_cz(params, wires):
    """Single-layer RY rotations followed by a CZ entangling chain."""
    if len(params) != len(wires):
        raise ValueError(f"RY-CZ expects one parameter per wire (got {len(params)} vs {len(wires)})")
    for w in range(len(wires)):
        qml.RY(params[w], wires=wires[w])
    for i in range(len(wires) - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])


def minimal(params, wires):
    """Minimal 2-qubit circuit: RY rotation + CNOT."""
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])


def hardware_efficient_ansatz(params, wires):
    """Standard hardware-efficient ansatz using StronglyEntanglingLayers."""
    qml.templates.StronglyEntanglingLayers(params, wires=wires)


# ================================================================
# CHEMISTRY-INSPIRED ANSATZES (UCC FAMILY)
# ================================================================
def uccsd_ansatz(params, wires, symbols=None, coordinates=None, basis="sto-3g"):
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    Uses single and double excitations built from a molecular Hartree–Fock reference.
    Results are cached per (molecule, geometry) for efficiency.
    """
    spin_orbitals = len(wires)

    # --- Build or reuse cached molecular data ---
    key = tuple(symbols) + tuple(coordinates.flatten())
    if not hasattr(uccsd_ansatz, "_cache"):
        uccsd_ansatz._cache = {}

    if key not in uccsd_ansatz._cache:
        try:
            mol = qchem.Molecule(symbols, coordinates, charge=0, basis=basis)
        except TypeError:
            # Older PennyLane compatibility
            mol = qchem.Molecule(symbols, coordinates, charge=0)

        electrons = mol.n_electrons
        singles, doubles = qchem.excitations(electrons, spin_orbitals)
        hf_state = qchem.hf_state(electrons, spin_orbitals)
        uccsd_ansatz._cache[key] = (singles, doubles, hf_state)
    else:
        singles, doubles, hf_state = uccsd_ansatz._cache[key]

    # --- Prepare reference state ---
    qml.BasisState(hf_state, wires=wires)

    # --- Apply single and double excitations ---
    n_singles = len(singles)
    for i, s in enumerate(singles):
        qml.SingleExcitation(params[i], wires=s)
    for j, d in enumerate(doubles):
        qml.DoubleExcitation(params[n_singles + j], wires=d)


def uccd_ansatz(params, wires, symbols=None, coordinates=None, basis="sto-3g"):
    """
    Unitary Coupled Cluster Doubles (UCCD) ansatz.
    Applies only double excitations (no singles). Used for LiH and similar systems.
    Cached molecular data to avoid repeated qchem rebuilds.
    """
    if symbols is None or coordinates is None:
        symbols = ["Li", "H"]
        coordinates = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.6]])

    spin_orbitals = len(wires)
    key = tuple(symbols) + tuple(coordinates.flatten())

    # --- Build or reuse cached molecular data ---
    if not hasattr(uccd_ansatz, "_cache"):
        uccd_ansatz._cache = {}

    if key not in uccd_ansatz._cache:
        try:
            mol = qchem.Molecule(symbols, coordinates, charge=0, basis=basis)
        except TypeError:
            mol = qchem.Molecule(symbols, coordinates, charge=0)

        electrons = mol.n_electrons
        _, doubles = qchem.excitations(electrons, spin_orbitals)
        hf_state = qchem.hf_state(electrons, spin_orbitals)
        uccd_ansatz._cache[key] = (doubles, hf_state)
    else:
        doubles, hf_state = uccd_ansatz._cache[key]

    # --- Prepare Hartree–Fock state ---
    qml.BasisState(hf_state, wires=wires)

    # --- Sanity check ---
    if len(params) != len(doubles):
        raise ValueError(f"UCCD expects {len(doubles)} parameters, got {len(params)}")

    # --- Apply double excitations ---
    for i, d in enumerate(doubles):
        qml.DoubleExcitation(params[i], wires=d)


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
}


def get_ansatz(name: str):
    """Return ansatz function by name."""
    if name not in ANSATZES:
        available = ", ".join(ANSATZES.keys())
        raise ValueError(f"Unknown ansatz '{name}'. Available: {available}")
    return ANSATZES[name]


# ================================================================
# PARAMETER INITIALIZATION
# ================================================================
def init_params(
    ansatz_name: str,
    num_wires: int,
    scale: float = 0.01,
    requires_grad=True,
    symbols=None,
    coordinates=None,
    basis="sto-3g",
    seed: int = 0,
):
    """
    Initialize variational parameters for a given ansatz.

    - Random initialization for hardware-efficient circuits.
    - Zero initialization for UCC-D/UCCD (per LiH_Noiseless.ipynb convention).
    - Deterministic for reproducibility (controlled by seed).

    Args:
        ansatz_name: Name of the ansatz.
        num_wires: Number of qubits.
        scale: Random scale for non-UCC ansatzes.
        requires_grad: Whether parameters require gradients.
        symbols, coordinates, basis: Molecular data (for UCC variants).
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Initialized parameter array.
    """
    np.random.seed(seed)

    if ansatz_name in ["TwoQubit-RY-CNOT", "Minimal"]:
        vals = scale * np.random.randn(1)

    elif ansatz_name == "RY-CZ":
        vals = scale * np.random.randn(num_wires)

    elif ansatz_name == "StronglyEntanglingLayers":
        vals = np.random.normal(0, np.pi, (1, num_wires, 3))

    elif ansatz_name in ["UCCSD", "UCC-D", "UCCD"]:
        # --- Default fallback geometry ---
        if symbols is None or coordinates is None:
            symbols = ["H", "H"]
            coordinates = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.7414]])

        spin_orbitals = num_wires
        try:
            mol = qchem.Molecule(symbols, coordinates, charge=0, basis=basis)
        except TypeError:
            mol = qchem.Molecule(symbols, coordinates, charge=0)

        electrons = mol.n_electrons
        singles, doubles = qchem.excitations(electrons, spin_orbitals)

        # --- Parameter allocation ---
        if ansatz_name in ["UCC-D", "UCCD"]:
            vals = np.zeros(len(doubles))
        else:
            vals = scale * np.random.randn(len(singles) + len(doubles))

    else:
        raise ValueError(f"Unknown ansatz '{ansatz_name}'")

    return np.array(vals, requires_grad=requires_grad)
