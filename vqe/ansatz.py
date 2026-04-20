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
    * UCCS      (singles only)
    * UCCD      (doubles only)
    * UCCSD     (singles + doubles)

All chemistry ansatzes are constructed to mirror the legacy
`excitation_ansatz(..., excitation_type=...)` behaviour from the old notebooks,
while keeping the interface compatible with `vqe.engine.build_ansatz(...)`.
"""

from __future__ import annotations

from typing import Any

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
            f"TwoQubit-RY-CNOT expects {len(wires) - 1} parameters for {len(wires)} wires, "
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
            f"RY-CZ expects one parameter per wire (got {len(params)} vs {len(wires)})"
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


def hardware_efficient_ansatz(params, wires, *, layers: int | None = None):
    """
    Standard hardware-efficient ansatz using StronglyEntanglingLayers.

    Convention:
        params.shape = (n_layers, len(wires), 3)
    """
    if layers is not None and int(layers) != int(np.shape(params)[0]):
        raise ValueError(
            "StronglyEntanglingLayers received params with "
            f"{int(np.shape(params)[0])} layers, but ansatz_kwargs requested "
            f"{int(layers)} layers."
        )
    qml.templates.StronglyEntanglingLayers(params, wires=wires)


def number_preserving_givens(params, wires, *, layers: int | None = None):
    """
    Nearest-neighbour number-preserving ansatz for lattice Hamiltonians.

    Each layer applies SingleExcitation rotations on nearest-neighbour bonds,
    which preserves the total excitation number prepared by reference_state.

    Convention:
        params.shape = (n_layers, len(wires) - 1)
    """
    wires = list(wires)
    if len(wires) < 2:
        raise ValueError("NumberPreservingGivens requires at least 2 wires.")

    params_arr = np.asarray(params)
    if params_arr.ndim == 1:
        params_arr = np.reshape(params_arr, (1, -1))

    expected_layers = int(layers) if layers is not None else int(params_arr.shape[0])
    expected_shape = (expected_layers, len(wires) - 1)
    if tuple(params_arr.shape) != expected_shape:
        raise ValueError(
            "NumberPreservingGivens expects params.shape == "
            f"{expected_shape}, got {tuple(params_arr.shape)}."
        )

    for layer in range(expected_layers):
        for bond in range(len(wires) - 1):
            qml.SingleExcitation(
                params_arr[layer, bond],
                wires=[wires[bond], wires[bond + 1]],
            )


def tfim_hamiltonian_variational(
    params,
    wires,
    *,
    layers: int | None = None,
    prepare_plus: bool = True,
):
    """
    Hamiltonian-variational ansatz for open-chain TFIM models.

    By default the circuit starts from |+...+>, then each layer applies
    nearest-neighbour ZZ entanglers followed by a transverse-field mixer:
        exp(-i beta_l sum_i X_i) exp(-i gamma_l sum_i Z_i Z_{i+1})

    Convention:
        params.shape = (n_layers, 2)
        params[:, 0] = beta_l
        params[:, 1] = gamma_l
    """
    wires = list(wires)
    if len(wires) < 2:
        raise ValueError("TFIM-HVA requires at least 2 wires.")

    params_arr = np.asarray(params)
    if params_arr.ndim == 1:
        params_arr = np.reshape(params_arr, (1, -1))

    expected_layers = int(layers) if layers is not None else int(params_arr.shape[0])
    expected_shape = (expected_layers, 2)
    if tuple(params_arr.shape) != expected_shape:
        raise ValueError(
            f"TFIM-HVA expects params.shape == {expected_shape}, "
            f"got {tuple(params_arr.shape)}."
        )

    if bool(prepare_plus):
        for wire in wires:
            qml.Hadamard(wires=wire)

    for layer in range(expected_layers):
        beta, gamma = params_arr[layer]
        for left, right in zip(wires[:-1], wires[1:]):
            qml.IsingZZ(2.0 * gamma, wires=[left, right])
        for wire in wires:
            qml.RX(2.0 * beta, wires=wire)


def xxz_hamiltonian_variational(params, wires, *, layers: int | None = None):
    """
    Hamiltonian-variational ansatz for open-chain XXZ/Heisenberg models.

    Each layer applies nearest-neighbour excitation-preserving exchange
    rotations and ZZ entanglers on even bonds, then odd bonds. The exchange
    block is implemented with ``SingleExcitation`` so the ansatz stays in the
    fixed-magnetization sector of the input reference state.

    Convention:
        params.shape = (n_layers, 4)
        params[:, 0] = theta_xy_even_l
        params[:, 1] = theta_z_even_l
        params[:, 2] = theta_xy_odd_l
        params[:, 3] = theta_z_odd_l
    """
    wires = list(wires)
    if len(wires) < 2:
        raise ValueError("XXZ-HVA requires at least 2 wires.")

    params_arr = np.asarray(params)
    if params_arr.ndim == 1:
        params_arr = np.reshape(params_arr, (1, -1))

    expected_layers = int(layers) if layers is not None else int(params_arr.shape[0])
    expected_shape = (expected_layers, 4)
    if tuple(params_arr.shape) != expected_shape:
        raise ValueError(
            f"XXZ-HVA expects params.shape == {expected_shape}, "
            f"got {tuple(params_arr.shape)}."
        )

    for layer in range(expected_layers):
        theta_xy_even, theta_z_even, theta_xy_odd, theta_z_odd = params_arr[layer]
        for start, theta_xy, theta_z in (
            (0, theta_xy_even, theta_z_even),
            (1, theta_xy_odd, theta_z_odd),
        ):
            for bond in range(start, len(wires) - 1, 2):
                left, right = wires[bond], wires[bond + 1]
                qml.SingleExcitation(theta_xy, wires=[left, right])
                qml.IsingZZ(2.0 * theta_z, wires=[left, right])


# ================================================================
# UCC-STYLE CHEMISTRY ANSATZES
# ================================================================
_UCC_DATA_CACHE: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}


def _ucc_cache_key(
    symbols,
    coordinates,
    basis: str,
    charge: int,
    multiplicity: int = 1,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
):
    """Build a hashable cache key from molecular data."""
    coords = np.array(coordinates, dtype=float).flatten().tolist()
    return (
        tuple(symbols),
        tuple(coords),
        basis.upper(),
        int(charge),
        int(multiplicity),
        None if active_electrons is None else int(active_electrons),
        None if active_orbitals is None else int(active_orbitals),
    )


def _build_ucc_data(
    symbols,
    coordinates,
    basis: str = "STO-3G",
    charge: int = 0,
    multiplicity: int = 1,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
):
    """
    Compute (singles, doubles, hf_state) for a given molecule and cache them.

    This mirrors the legacy notebook logic based on:
        - qchem.hf_state(electrons, spin_orbitals)
        - qchem.excitations(electrons, spin_orbitals)

    Notes
    -----
    * The module-level cache keeps repeated calls cheap.
    """
    if symbols is None or coordinates is None:
        raise ValueError(
            "UCC ansatz requires symbols and coordinates. "
            "Make sure build_hamiltonian(...) is used and passed through."
        )

    charge_int = int(charge)
    key = _ucc_cache_key(
        symbols,
        coordinates,
        basis,
        charge_int,
        int(multiplicity),
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    if key not in _UCC_DATA_CACHE:
        try:
            mol = qchem.Molecule(
                symbols,
                coordinates,
                charge=charge_int,
                mult=int(multiplicity),
                basis_name=basis,
            )
        except TypeError:
            try:
                mol = qchem.Molecule(
                    symbols,
                    coordinates,
                    charge=charge_int,
                    mult=int(multiplicity),
                    basis=basis,
                )
            except TypeError:
                mol = qchem.Molecule(
                    symbols,
                    coordinates,
                    charge=charge_int,
                    mult=int(multiplicity),
                )

        if active_electrons is None and active_orbitals is None:
            electrons = int(mol.n_electrons)
            spin_orbitals = 2 * int(mol.n_orbitals)
        else:
            core, active = qchem.active_space(
                int(mol.n_electrons),
                int(mol.n_orbitals),
                mult=int(multiplicity),
                active_electrons=(
                    None if active_electrons is None else int(active_electrons)
                ),
                active_orbitals=(
                    None if active_orbitals is None else int(active_orbitals)
                ),
            )
            electrons = int(mol.n_electrons) - 2 * len(core)
            spin_orbitals = 2 * len(active)

        singles, doubles = qchem.excitations(electrons, spin_orbitals)
        hf_state = qchem.hf_state(electrons, spin_orbitals)

        singles = [tuple(ex) for ex in singles]
        doubles = [tuple(ex) for ex in doubles]
        hf_state = np.array(hf_state, dtype=int)

        _UCC_DATA_CACHE[key] = (singles, doubles, hf_state)

    return _UCC_DATA_CACHE[key]


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
    charge: int = 0,
    multiplicity: int = 1,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
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
    singles, doubles, hf_state = _build_ucc_data(
        symbols,
        coordinates,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

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
    charge: int = 0,
    multiplicity: int = 1,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    reference_state=None,
    prepare_reference: bool = True,
):
    """
    UCCD: doubles-only UCC ansatz.

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
    singles, doubles, hf_state = _build_ucc_data(
        symbols,
        coordinates,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

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
    charge: int = 0,
    multiplicity: int = 1,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    reference_state=None,
    prepare_reference: bool = True,
):
    """
    UCCS: singles-only UCC ansatz.

    Matches the structure of UCCSD/UCCD and the legacy
    `excitation_ansatz(..., excitation_type="single")` behaviour.
    """
    singles, doubles, hf_state = _build_ucc_data(
        symbols,
        coordinates,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

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
    "NumberPreservingGivens": number_preserving_givens,
    "TFIM-HVA": tfim_hamiltonian_variational,
    "XXZ-HVA": xxz_hamiltonian_variational,
    "UCCSD": uccsd_ansatz,
    "UCCD": uccd_ansatz,
    "UCCS": uccs_ansatz,
}

_ANSATZ_ALIASES = {
    "UCC-SD": "UCCSD",
    "UCC-D": "UCCD",
    "UCC-S": "UCCS",
}


def _normalize_ansatz_key(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch not in " _-")


def canonicalize_ansatz_name(name: str) -> str:
    """Map case/spacing variants and legacy aliases to canonical registry names."""
    normalized = str(name).strip()
    normalized_key = _normalize_ansatz_key(normalized)

    lookup = {
        _normalize_ansatz_key("TwoQubit-RY-CNOT"): "TwoQubit-RY-CNOT",
        _normalize_ansatz_key("RY-CZ"): "RY-CZ",
        _normalize_ansatz_key("Minimal"): "Minimal",
        _normalize_ansatz_key("StronglyEntanglingLayers"): "StronglyEntanglingLayers",
        _normalize_ansatz_key("NumberPreservingGivens"): "NumberPreservingGivens",
        _normalize_ansatz_key("SSH-Givens"): "NumberPreservingGivens",
        _normalize_ansatz_key("TFIM-HVA"): "TFIM-HVA",
        _normalize_ansatz_key("TFIMHamiltonianVariational"): "TFIM-HVA",
        _normalize_ansatz_key("XXZ-HVA"): "XXZ-HVA",
        _normalize_ansatz_key("Heisenberg-HVA"): "XXZ-HVA",
        _normalize_ansatz_key("XXZHamiltonianVariational"): "XXZ-HVA",
        _normalize_ansatz_key("UCCSD"): "UCCSD",
        _normalize_ansatz_key("UCCD"): "UCCD",
        _normalize_ansatz_key("UCCS"): "UCCS",
        _normalize_ansatz_key("UCC-SD"): "UCCSD",
        _normalize_ansatz_key("UCC-D"): "UCCD",
        _normalize_ansatz_key("UCC-S"): "UCCS",
    }

    if normalized in _ANSATZ_ALIASES:
        return _ANSATZ_ALIASES[normalized]
    return lookup.get(normalized_key, normalized)


def get_ansatz(name: str):
    """
    Return ansatz function by name.

    This is the entry point used by `vqe.engine.build_ansatz(...)`.
    """
    name = canonicalize_ansatz_name(name)

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
    charge: int = 0,
    multiplicity: int = 1,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    ansatz_kwargs: dict[str, Any] | None = None,
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

    - UCC family (UCCS / UCCD / UCCSD)
        * **All zeros**, starting from θ = 0 as in the original chemistry notebooks.
          The length of the vector is determined from the excitation lists.

    Returns
    -------
    np.ndarray
        Parameter array with `requires_grad=True`
    """
    np.random.seed(seed)

    ansatz_name = canonicalize_ansatz_name(ansatz_name)
    options = dict(ansatz_kwargs or {})

    def _positive_int_option(name: str, default: int) -> int:
        value = int(options.get(name, default))
        if value < 1:
            raise ValueError(f"ansatz_kwargs['{name}'] must be >= 1.")
        return value

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
        # n layers, 3 parameters per wire
        layers = _positive_int_option("layers", 1)
        vals = np.random.normal(loc=0.0, scale=np.pi, size=(layers, num_wires, 3))

    elif ansatz_name == "NumberPreservingGivens":
        if num_wires < 2:
            raise ValueError("NumberPreservingGivens requires at least 2 wires.")
        layers = _positive_int_option("layers", 1)
        vals = scale * np.random.randn(layers, num_wires - 1)

    elif ansatz_name in {"TFIM-HVA", "XXZ-HVA"}:
        if num_wires < 2:
            raise ValueError(f"{ansatz_name} requires at least 2 wires.")
        layers = _positive_int_option("layers", 1)
        params_per_layer = 2 if ansatz_name == "TFIM-HVA" else 4
        vals = scale * np.random.randn(layers, params_per_layer)

    elif ansatz_name in ["UCCSD", "UCCD", "UCCS"]:
        if symbols is None or coordinates is None:
            raise ValueError(
                f"Ansatz '{ansatz_name}' requires symbols/coordinates "
                "to determine excitation count. Ensure you are using "
                "build_hamiltonian(...) and engine.build_ansatz(...)."
            )

        singles, doubles, _ = _build_ucc_data(
            symbols,
            coordinates,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
            active_electrons=active_electrons,
            active_orbitals=active_orbitals,
        )

        if ansatz_name == "UCCD":
            # doubles-only
            vals = np.zeros(len(doubles))

        elif ansatz_name == "UCCS":
            # singles-only
            vals = np.zeros(len(singles))

        else:
            # UCCSD: singles + doubles
            vals = np.zeros(len(singles) + len(doubles))

    else:
        available = ", ".join(sorted(ANSATZES.keys()))
        raise ValueError(f"Unknown ansatz '{ansatz_name}'. Available: {available}")

    return np.array(vals, requires_grad=requires_grad)
