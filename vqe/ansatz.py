import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem


def two_qubit_ry_cnot(params, wires):
    """Toy 2-qubit entangler; NOT chemical UCCSD."""
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(-params[0], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])


def ry_cz(params, wires):
    """RY rotations followed by CZ entanglement."""
    if len(params) != len(wires):
        raise ValueError("ry_cz expects one parameter per wire")
    for w in range(len(wires)):
        qml.RY(params[w], wires=wires[w])
    for i in range(len(wires) - 1):
        qml.CZ(wires=[wires[i], wires[i + 1]])


def minimal(params, wires):
    """Minimal 2-qubit circuit."""
    qml.RY(params[0], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])


def hardware_efficient_ansatz(params, wires):
    """Default hardware-efficient ansatz."""
    qml.templates.StronglyEntanglingLayers(params, wires=wires)


# 🧬 UCCSD ansatz definition (must come BEFORE ANSATZES)
# inside uccsd_ansatz
def uccsd_ansatz(params, wires, symbols=None, coordinates=None, basis="sto-3g"):
    """UCCSD-style ansatz using singles and doubles excitations."""
    spin_orbitals = len(wires)

    # Avoid rebuilding molecule if it's already computed once per run
    key = tuple(symbols) + tuple(coordinates.flatten())
    if not hasattr(uccsd_ansatz, "_cache"):
        uccsd_ansatz._cache = {}
    if key not in uccsd_ansatz._cache:
        try:
            mol = qchem.Molecule(symbols, coordinates, charge=0, basis=basis)
        except TypeError:
            mol = qchem.Molecule(symbols, coordinates, charge=0)
        electrons = mol.n_electrons
        singles, doubles = qchem.excitations(electrons, spin_orbitals)
        hf_state = qchem.hf_state(electrons, spin_orbitals)
        uccsd_ansatz._cache[key] = (singles, doubles, hf_state)
    else:
        singles, doubles, hf_state = uccsd_ansatz._cache[key]

    qml.BasisState(hf_state, wires=wires)

    n_singles = len(singles)
    n_doubles = len(doubles)
    for i, s in enumerate(singles):
        qml.SingleExcitation(params[i], wires=s)
    for j, d in enumerate(doubles):
        qml.DoubleExcitation(params[n_singles + j], wires=d)


def uccd_ansatz(params, wires, symbols=None, coordinates=None, basis="sto-3g"):
    """UCC Doubles (UCCD) ansatz — only double excitations applied."""
    if symbols is None or coordinates is None:
        symbols = ["Li", "H"]
        coordinates = np.array([[0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.6]])

    spin_orbitals = len(wires)

    try:
        mol = qchem.Molecule(symbols, coordinates, charge=0)
    except TypeError:
        mol = qchem.Molecule(symbols, coordinates, charge=0, basis=basis)

    electrons = mol.n_electrons
    _, doubles = qchem.excitations(electrons, spin_orbitals)
    hf_state = qchem.hf_state(electrons, spin_orbitals)
    qml.BasisState(hf_state, wires=wires)

    if len(params) != len(doubles):
        raise ValueError(f"UCCD expects {len(doubles)} parameters; got {len(params)}")

    for i, d in enumerate(doubles):
        qml.DoubleExcitation(params[i], wires=d)


# --- registry of available ansatz functions ---
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
        raise ValueError(f"Ansatz '{name}' not recognized. Available: {list(ANSATZES.keys())}")
    return ANSATZES[name]


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
    """Initialize parameters based on ansatz structure, including UCCSD and UCC-D."""
    
    np.random.seed(seed)

    if ansatz_name in ["TwoQubit-RY-CNOT", "Minimal"]:
        vals = scale * np.random.randn(1)

    elif ansatz_name == "RY-CZ":
        vals = scale * np.random.randn(num_wires)

    elif ansatz_name == "StronglyEntanglingLayers":
        vals = np.random.normal(0, np.pi, (1, num_wires, 3))

    elif ansatz_name in ["UCCSD", "UCC-D", "UCCD"]:
        from pennylane import qchem

        # Default fallback for missing molecule info
        if symbols is None or coordinates is None:
            symbols = ["H", "H"]
            coordinates = np.array([[0.0, 0.0, 0.0],
                                    [0.0, 0.0, 0.7414]])

        spin_orbitals = num_wires

        try:
            mol = qchem.Molecule(symbols, coordinates, charge=0, basis=basis)
        except TypeError:
            # For older PennyLane versions
            mol = qchem.Molecule(symbols, coordinates, charge=0)

        electrons = mol.n_electrons

        # Compute excitations within the *same orbital space* as Hamiltonian
        singles, doubles = qchem.excitations(electrons, spin_orbitals)

        if ansatz_name in ["UCC-D", "UCCD"]:
            n_params = len(doubles)
        else:
            n_params = len(singles) + len(doubles)

        vals = scale * np.random.randn(n_params)

    else:
        raise ValueError(f"Unknown ansatz '{ansatz_name}'")

    return np.array(vals, requires_grad=requires_grad)
