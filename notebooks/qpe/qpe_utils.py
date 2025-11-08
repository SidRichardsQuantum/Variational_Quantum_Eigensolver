import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import os, json, hashlib, time
import matplotlib.pyplot as plt
from notebooks.vqe.vqe_utils import set_seed

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "qpe_results")
IMG_DIR = os.path.join(ROOT_DIR, "plots", "qpe")

def ensure_dirs():
    """Ensure required directories exist (parallel to VQE)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)


# Noise Model
def apply_noise_all(wires, p_dep=0.0, p_amp=0.0):
    """Apply depolarizing / amplitude damping noise to all given wires."""
    for w in wires:
        if p_dep > 0.0:
            qml.DepolarizingChannel(p_dep, wires=w)
        if p_amp > 0.0:
            qml.AmplitudeDamping(p_amp, wires=w)


# Inverse Quantum Fourier Transform
def inverse_qft(wires):
    """IQFT on provided wires."""
    n = len(wires)
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - i - 1]])
    for j in range(n):
        k = n - j - 1
        qml.Hadamard(wires=k)
        for m in range(k):
            angle = -np.pi / (2 ** (k - m))
            qml.ControlledPhaseShift(angle, wires=[wires[m], wires[k]])


# Controlled Powered Evolution
def controlled_powered_evolution(
    hamiltonian,
    system_wires,
    control_wire,
    t,
    power,
    trotter_steps=1,
    noise_params=None,
):
    """
    Apply controlled-U^(2^power) with optional noise.
    U = exp(-i H t)
    """
    n_repeat = 2 ** power
    for _ in range(n_repeat):
        qml.ctrl(qml.ApproxTimeEvolution, control=control_wire)(
            hamiltonian, t, trotter_steps, system_wires)
        if noise_params:
            apply_noise_all(
                wires=system_wires + [control_wire],
                p_dep=noise_params.get("p_dep", 0.0),
                p_amp=noise_params.get("p_amp", 0.0),
            )


# Run Quantum Phase Estimation
def run_qpe(
    hamiltonian,
    hf_state,
    n_ancilla=4,
    t=1.0,
    trotter_steps=1,
    noise_params=None,
    shots=5000,
    molecule_name="molecule",
):
    """
    Run QPE (with optional noise) and return results.
    Mirrors run_vqe() style in vqe_utils.py.
    """
    num_qubits = len(hf_state)
    ancilla_wires = list(range(n_ancilla))
    system_wires = list(range(n_ancilla, n_ancilla + num_qubits))

    # Device
    dev = qml.device("default.mixed", wires=n_ancilla + num_qubits, shots=shots)

    # Map the Hamiltonian onto the system wires used in the QPE circuit
    H_sys = hamiltonian.map_wires(dict(zip(range(num_qubits), system_wires)))

    # QPE Circuit
    @qml.qnode(dev)
    def circuit():
        # Prepare HF on system
        qml.BasisState(np.array(hf_state, dtype=int), wires=system_wires)

        # Hadamards on ancilla
        for a in ancilla_wires:
            qml.Hadamard(wires=a)

        # Controlled U^(2^(n_ancilla-1-k))
        for k, a in enumerate(ancilla_wires):
            n_repeat = 2 ** (n_ancilla - 1 - k)
            for _ in range(n_repeat):
                qml.ctrl(qml.ApproxTimeEvolution, control=a)(H_sys, t, trotter_steps)
                if noise_params:
                    apply_noise_all(
                        wires=system_wires + [a],
                        p_dep=noise_params.get("p_dep", 0.0),
                        p_amp=noise_params.get("p_amp", 0.0),
                    )

        # Inverse QFT on ancillas and sample them
        inverse_qft(ancilla_wires)
        return qml.sample(wires=ancilla_wires)

    # Execute circuit
    samples = np.array(circuit(), dtype=int)

    # Build bitstrings (MSB→LSB, i.e., ancilla_wires order as prepared)
    bitstrings = ["".join(str(int(b)) for b in s) for s in samples]

    # Tallies
    counts = {}
    for b in bitstrings:
        counts[b] = counts.get(b, 0) + 1

    probs = {b: c / shots for b, c in counts.items()}

    # HF reference energy on the ORIGINAL wire labels (remap back)
    E_hf = hartree_fock_energy(
        H_sys.map_wires(dict(zip(system_wires, range(num_qubits)))), hf_state
    )

    # Build rows with both MSB/LSB interpretations
    rows = []
    for b, c in counts.items():
        ph_m = bitstring_to_phase(b, msb_first=True)
        ph_l = bitstring_to_phase(b, msb_first=False)
        e_m = phase_to_energy_unwrapped(ph_m, t, ref_energy=E_hf)
        e_l = phase_to_energy_unwrapped(ph_l, t, ref_energy=E_hf)
        rows.append((b, c, ph_m, ph_l, e_m, e_l))

    # Choose most probable bitstring, then pick energy (MSB/LSB) closest to HF
    best_row = max(rows, key=lambda r: r[1])  # r = (b, count, ph_m, ph_l, e_m, e_l)
    best_b = best_row[0]
    best_E = min((best_row[4], best_row[5]), key=lambda x: abs(x - E_hf))
    best_phase = best_row[2] if best_E == best_row[4] else best_row[3]

    result = {
        "counts": counts,
        "probs": probs,
        "best_bitstring": best_b,
        "phase": float(best_phase),
        "energy": float(best_E),
        "n_ancilla": n_ancilla,
        "t": t,
        "noise": noise_params,
        "shots": shots,
        "molecule": molecule_name,
    }

    return result


def bits_to_phase(bitstring):
    """Convert binary string (MSB→LSB) to phase ∈ [0,1)."""
    phase = 0.0
    for i, b in enumerate(bitstring[::-1]):
        phase += int(b) / (2 ** (i + 1))
    return phase

def phase_to_energy(phase, t):
    """Compute energy (Ha) from measured phase."""
    E = 2 * np.pi * phase / t
    if E > np.pi / t:
        E -= 2 * np.pi / t
    return E


# Cashe results
def signature_hash(molecule, n_ancilla, t, noise, shots):
    """Generate a reproducible hash key for caching results."""
    signature = json.dumps(
        {"mol": molecule, "anc": n_ancilla, "t": t, "noise": noise, "shots": shots},
        sort_keys=True,
    )
    return hashlib.md5(signature.encode()).hexdigest()

def cache_path(molecule, hash_key):
    """Build cache file path."""
    return os.path.join(DATA_DIR, f"{molecule}_QPE_{hash_key}.json")

def save_qpe_result(result):
    """Save QPE result to JSON (parallels save_vqe_result)."""
    ensure_dirs()
    key = signature_hash(
        result["molecule"],
        result["n_ancilla"],
        result["t"],
        result["noise"],
        result["shots"],
    )
    path = cache_path(result["molecule"], key)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved QPE result → {path}")

def load_qpe_result(molecule, hash_key):
    """Load cached QPE result if available."""
    path = cache_path(molecule, hash_key)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


# Plotting QPE Results
def plot_qpe_distribution(result):
    """Bar plot of ancilla probability distribution (in ket notation)."""
    probs = result["probs"]
    items = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    xs = [f"|{b}⟩" for b, _ in items]
    ys = [p for _, p in items]

    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys)
    plt.xlabel("Ancilla register state")
    plt.ylabel("Probability")
    plt.title(
        f"QPE Phase Distribution – {result['molecule']} "
        f"(ancillas={result['n_ancilla']}, noise={result['noise']})"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    fname = os.path.join(
        IMG_DIR, f"{result['molecule']}_QPE_{result['n_ancilla']}q.png"
    )
    plt.savefig(fname, dpi=200)
    plt.show()
    print(f"Saved plot → {fname}")


# Hartree–Fock Energy Comparison
def hartree_fock_energy(hamiltonian, hf_state):
    """Compute ⟨HF|H|HF⟩ for reference energy (Ha)."""
    num_qubits = len(hf_state)
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(hf_state, wires=range(num_qubits))
        return qml.expval(hamiltonian)

    return circuit()


def bitstring_to_phase(bits: str, msb_first: bool = True) -> float:
    """Convert a 0/1 string to fractional phase in [0,1)."""
    b = bits if msb_first else bits[::-1]
    frac = 0.0
    for i, ch in enumerate(b, start=1):
        frac += (ch == "1") * (0.5 ** i)
    return float(frac)


def phase_to_energy_unwrapped(phase: float, t: float, ref_energy: float | None = None) -> float:
    """Convert phase → energy and unwrap sign convention to match physics."""
    # Base (always positive for phase in [0,1))
    base = 2 * np.pi * phase / t

    # Invert sign
    energy = -base

    # Bring into (-π/t, π/t]
    if energy > np.pi / t:
        energy -= 2 * np.pi / t
    elif energy <= -np.pi / t:
        energy += 2 * np.pi / t

    # If a reference is given, pick branch closest to it
    if ref_energy is not None:
        candidates = [energy + k * (2 * np.pi / t) for k in (-1, 0, 1)]
        energy = min(candidates, key=lambda x: abs(x - ref_energy))

    return energy


def save_qpe_plot(name: str):
    """Return full image path in the QPE plots directory and ensure directory exists."""
    ensure_dirs()
    return os.path.join(IMG_DIR, f"{name}")
