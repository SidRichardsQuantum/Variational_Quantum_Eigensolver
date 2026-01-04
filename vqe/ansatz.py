"""
vqe.ansatz
----------
Library of parameterized quantum circuits (ansatzes) used in the VQE workflow.

Includes
--------
- Simple 2-qubit toy ansatzes:
    * TwoQubit-RY-CNOT
    * Minimal
    * RY-CZ
- Hardware-efficient template:
    * StronglyEntanglingLayers
- Chemistry-inspired UCC family:
    * UCC-S     (singles only)
    * UCC-D     (doubles only)
    * UCCSD     (singles + doubles)

All chemistry ansatzes are constructed to mirror the legacy
`excitation_ansatz(..., excitation_type=...)` behaviour from the old notebooks,
while keeping the interface compatible with `vqe.engine.build_ansatz(...)`.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem


# ================================================================
# BASIC / TOY ANSATZES
# ================================================================
def two_qubit_ry_cnot(params, wires):
    """
    Scalable version of the original 2-qubit RY-CNOT motif.

    Applies the motif to every adjacent pair of qubits:
        RY(param) on wire i
        CNOT(i → i+1)
        RY(-param) on wire i+1
        CNOT(i → i+1)

    Number of parameters = len(wires) - 1.
    """
    if len(params) != len(wires) - 1:
        raise ValueError(
            f"TwoQubit-RY-CNOT expects {len(wires)-1} parameters for {len(wires)} wires, "
            f"got {len(params)}."
        )

    for i in range(len(wires) - 1):
        w0, w1 = wires[i], wires[i + 1]
        theta = params[i]

        qml.RY(theta, wires=w0)
        qml.CNOT(wires=[w0, w1])
        qml.RY(-theta, wires=w1)
        qml.CNOT(wires=[w0, w1])


def ry_cz(params, wires):
    """
    Single-layer RY rotations followed by a CZ chain.

    Matches the legacy `vqe_utils.ry_cz` used in H₂ optimizer / ansatz
    comparison notebooks.

    Shape:
        params.shape == (len(wires),)
    """
    if len(params) != len(wires):
        raise ValueError(
            f"RY-CZ expects one parameter per wire "
            f"(got {len(params)} vs {len(wires)})"
        )

    # Local rotations
    for theta, w in zip(params, wires):
        qml.RY(theta, wires=w)

    # Entangling CZ chain
    for w0, w1 in zip(wires[:-1], wires[1:]):
        qml.CZ(wires=[w0, w1])


def minimal(params, wires):
    """
    Minimal 2-qubit circuit: RY rotation + CNOT.

    Matches the legacy vqe_utils.minimal used in H₂ ansatz comparisons.

    Behaviour:
        - Uses the first two wires from the provided wire list.
        - Requires at least 2 wires, but can be embedded in a larger register.
    """
    if len(wires) < 2:
        raise ValueError(f"Minimal ansatz expects at least 2 wires, got {len(wires)}")

    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])


def hardware_efficient_ansatz(params, wires):
    """
    Standard hardware-efficient ansatz using StronglyEntanglingLayers.

    Convention:
        params.shape = (n_layers, len(wires), 3)
    """
    qml.templates.StronglyEntanglingLayers(params, wires=wires)


# ================================================================
# UCC-STYLE CHEMISTRY ANSATZES
# ================================================================
def _ucc_cache_key(symbols, coordinates, basis: str):
    """Build a hashable cache key from molecular data."""
    coords = np.array(coordinates, dtype=float).flatten().tolist()
    return (tuple(symbols), tuple(coords), basis.upper())


def _build_ucc_data(symbols, coordinates, basis: str = "STO-3G"):
    """
    Compute (singles, doubles, hf_state) for a given molecule and cache them.

    This mirrors the legacy notebook logic based on:
        - qchem.hf_state(electrons, spin_orbitals)
        - qchem.excitations(electrons, spin_orbitals)

    Notes
    -----
    * We intentionally keep the call signature minimal: (symbols, coordinates, basis)
      so that `vqe.engine.build_ansatz(...)` can pass through the values it has
      without needing charge / multiplicity.
    * The cache lives on the function object so repeated calls are cheap.
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
            mol = qchem.Molecule(symbols, coordinates, charge=0)

        electrons = mol.n_electrons
        spin_orbitals = 2 * mol.n_orbitals

        singles, doubles = qchem.excitations(electrons, spin_orbitals)
        hf_state = qchem.hf_state(electrons, spin_orbitals)

        singles = [tuple(ex) for ex in singles]
        doubles = [tuple(ex) for ex in doubles]
        hf_state = np.array(hf_state, dtype=int)

        _build_ucc_data._cache[key] = (singles, doubles, hf_state)

    return _build_ucc_data._cache[key]


def _apply_ucc_layers(
    params,
    wires,
    *,
    singles,
    doubles,
    hf_state,
    use_singles: bool,
    use_doubles: bool,
    reference_state=None,
    prepare_reference: bool = True,
):
    """
    Shared helper to apply HF preparation + selected UCC excitation layers.

    Parameter ordering convention (matches legacy notebooks):
        - singles parameters first (if used)
        - doubles parameters after that
    """
    wires = list(wires)
    num_wires = len(wires)

    if len(hf_state) != num_wires:
        raise ValueError(
            f"HF state length ({len(hf_state)}) does not match number of wires "
            f"({num_wires})."
        )

    # Reference preparation
    # - If prepare_reference=False: assume caller has already prepared a state.
    # - Else if reference_state is provided: prepare that basis state.
    # - Else: prepare Hartree–Fock reference (default / legacy behavior).
    if prepare_reference:
        if reference_state is not None:
            ref = np.array(reference_state, dtype=int)
            if len(ref) != num_wires:
                raise ValueError(
                    f"reference_state length ({len(ref)}) does not match "
                    f"number of wires ({num_wires})."
                )
            qml.BasisState(ref, wires=wires)
        else:
            qml.BasisState(hf_state, wires=wires)

    # Determine how many parameters we expect
    n_singles = len(singles) if use_singles else 0
    n_doubles = len(doubles) if use_doubles else 0
    expected = n_singles + n_doubles

    if len(params) != expected:
        raise ValueError(
            f"UCC ansatz expects {expected} parameters, got {len(params)}."
        )

    # Apply singles
    offset = 0
    if use_singles:
        for i, exc in enumerate(singles):
            qml.SingleExcitation(params[offset + i], wires=list(exc))
        offset += n_singles

    # Apply doubles
    if use_doubles:
        for j, exc in enumerate(doubles):
            qml.DoubleExcitation(params[offset + j], wires=list(exc))


def uccsd_ansatz(
    params,
    wires,
    *,
    symbols=None,
    coordinates=None,
    basis: str = "STO-3G",
    reference_state=None,
    prepare_reference: bool = True,
):
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    Behaviour is chosen to match the legacy usage:

        excitation_ansatz(
            params,
            wires=range(qubits),
            hf_state=hf,
            excitations=(singles, doubles),
            excitation_type="both",
        )

    Args
    ----
    params
        1D array of length len(singles) + len(doubles)
    wires
        Sequence of qubit wires
    symbols, coordinates, basis
        Molecular information (must be provided for chemistry simulations)
    """
    singles, doubles, hf_state = _build_ucc_data(symbols, coordinates, basis=basis)

    _apply_ucc_layers(
        params,
        wires=wires,
        singles=singles,
        doubles=doubles,
        hf_state=hf_state,
        use_singles=True,
        use_doubles=True,
        reference_state=reference_state,
        prepare_reference=prepare_reference,
    )


def uccd_ansatz(
    params,
    wires,
    *,
    symbols=None,
    coordinates=None,
    basis: str = "STO-3G",
    reference_state=None,
    prepare_reference: bool = True,
):
    """
    UCC-D / UCCD: doubles-only UCC ansatz.

    Designed to mirror the LiH notebook behaviour where we used
    `excitation_ansatz(..., excitation_type="double")` with zero initial params.

    Args
    ----
    params
        1D array of length len(doubles)
    wires
        Sequence of qubit wires
    symbols, coordinates, basis
        Molecular information
    """
    singles, doubles, hf_state = _build_ucc_data(symbols, coordinates, basis=basis)

    _apply_ucc_layers(
        params,
        wires=wires,
        singles=singles,
        doubles=doubles,
        hf_state=hf_state,
        use_singles=False,
        use_doubles=True,
        reference_state=reference_state,
        prepare_reference=prepare_reference,
    )


def uccs_ansatz(
    params,
    wires,
    *,
    symbols=None,
    coordinates=None,
    basis: str = "STO-3G",
    reference_state=None,
    prepare_reference: bool = True,
):
    """
    UCC-S: singles-only UCC ansatz.

    Matches the structure of UCCSD/UCCD and the legacy
    `excitation_ansatz(..., excitation_type="single")` behaviour.
    """
    singles, doubles, hf_state = _build_ucc_data(symbols, coordinates, basis=basis)

    _apply_ucc_layers(
        params,
        wires=wires,
        singles=singles,
        doubles=doubles,
        hf_state=hf_state,
        use_singles=True,
        use_doubles=False,
        reference_state=reference_state,
        prepare_reference=prepare_reference,
    )


# ================================================================
# REGISTRY
# ================================================================
ANSATZES = {
    "TwoQubit-RY-CNOT": two_qubit_ry_cnot,
    "RY-CZ": ry_cz,
    "Minimal": minimal,
    "StronglyEntanglingLayers": hardware_efficient_ansatz,
    "UCCSD": uccsd_ansatz,
    "UCC-SD": uccsd_ansatz,  # alias
    "UCC-D": uccd_ansatz,
    "UCCD": uccd_ansatz,  # alias
    "UCC-S": uccs_ansatz,
    "UCCS": uccs_ansatz,  # alias
}


def get_ansatz(name: str):
    """
    Return ansatz function by name.

    This is the entry point used by `vqe.engine.build_ansatz(...)`.
    """
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

    Design choices (kept consistent with the legacy notebooks):

    - TwoQubit-RY-CNOT / Minimal
        * 1 parameter, small random normal ~ N(0, scale²)

    - RY-CZ
        * `num_wires` parameters, random normal ~ N(0, scale²)

    - StronglyEntanglingLayers
        * params.shape = (1, num_wires, 3), normal with width ~ π

    - UCC family (UCC-S / UCC-D / UCCSD and aliases)
        * **All zeros**, starting from θ = 0 as in the original chemistry notebooks.
          The length of the vector is determined from the excitation lists.

    Returns
    -------
    np.ndarray
        Parameter array with `requires_grad=True`
    """
    np.random.seed(seed)

    ansatz_name = str(ansatz_name).strip()

    # --- Toy ansatzes --------------------------------------------------------
    if ansatz_name == "TwoQubit-RY-CNOT":
        # scalable: one parameter per adjacent pair
        if num_wires < 2:
            raise ValueError("TwoQubit-RY-CNOT requires at least 2 wires.")
        vals = scale * np.random.randn(num_wires - 1)

    elif ansatz_name == "Minimal":
        # still a 1-parameter global circuit
        vals = scale * np.random.randn(1)

    elif ansatz_name == "RY-CZ":
        vals = scale * np.random.randn(num_wires)

    # --- Chemistry ansatzes (UCC family) ------------------------------------
    elif ansatz_name == "StronglyEntanglingLayers":
        # 1 layer, 3 parameters per wire
        vals = np.random.normal(loc=0.0, scale=np.pi, size=(1, num_wires, 3))

    elif ansatz_name in ["UCCSD", "UCC-SD", "UCC-D", "UCCD", "UCC-S", "UCCS"]:
        if symbols is None or coordinates is None:
            raise ValueError(
                f"Ansatz '{ansatz_name}' requires symbols/coordinates "
                "to determine excitation count. Ensure you are using "
                "build_hamiltonian(...) and engine.build_ansatz(...)."
            )

        singles, doubles, _ = _build_ucc_data(symbols, coordinates, basis=basis)

        if ansatz_name in ["UCC-D", "UCCD"]:
            # doubles-only
            vals = np.zeros(len(doubles))

        elif ansatz_name in ["UCC-S", "UCCS"]:
            # singles-only
            vals = np.zeros(len(singles))

        else:
            # UCCSD / UCC-SD: singles + doubles
            vals = np.zeros(len(singles) + len(doubles))

    else:
        available = ", ".join(sorted(ANSATZES.keys()))
        raise ValueError(f"Unknown ansatz '{ansatz_name}'. Available: {available}")

    return np.array(vals, requires_grad=requires_grad)
