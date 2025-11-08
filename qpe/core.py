"""
QPE core module.
Implements noiseless and noisy QPE using PennyLane,
mirroring the structure of the VQE package.
"""

import pennylane as qml
from pennylane import numpy as np
from typing import Dict, Any, Optional
from qpe.noise import apply_noise_all


# ---------------------------------------------------------------------
# Inverse Quantum Fourier Transform
# ---------------------------------------------------------------------
def inverse_qft(wires):
    """Apply inverse QFT to the provided wires."""
    n = len(wires)
    # Swap qubits to reverse order
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - i - 1]])
    # Apply controlled phase shifts and Hadamards
    for j in range(n):
        k = n - j - 1
        qml.Hadamard(wires=k)
        for m in range(k):
            angle = -np.pi / (2 ** (k - m))
            qml.ControlledPhaseShift(angle, wires=[wires[m], wires[k]])


# ---------------------------------------------------------------------
# Controlled Powered Evolution
# ---------------------------------------------------------------------
def controlled_powered_evolution(
    hamiltonian,
    system_wires,
    control_wire,
    t,
    power,
    trotter_steps=1,
    noise_params: Optional[Dict[str, float]] = None,
):
    """Apply controlled-U^(2^power) with optional noise.
    U = exp(-i * H * t)
    """
    n_repeat = 2 ** power
    for _ in range(n_repeat):
        qml.ctrl(qml.ApproxTimeEvolution, control=control_wire)(
            hamiltonian, t, trotter_steps, system_wires
        )
        if noise_params:
            apply_noise_all(
                wires=system_wires + [control_wire],
                p_dep=noise_params.get("p_dep", 0.0),
                p_amp=noise_params.get("p_amp", 0.0),
            )


# ---------------------------------------------------------------------
# QPE Runner
# ---------------------------------------------------------------------
def run_qpe(
    hamiltonian,
    hf_state,
    n_ancilla=4,
    t=1.0,
    trotter_steps=1,
    noise_params: Optional[Dict[str, float]] = None,
    shots=5000,
    molecule_name: str = "molecule",
) -> Dict[str, Any]:
    """Run QPE (noisy or noiseless) and return measurement results."""
    num_qubits = len(hf_state)
    ancilla_wires = list(range(n_ancilla))
    system_wires = list(range(n_ancilla, n_ancilla + num_qubits))

    dev = qml.device("default.mixed", wires=n_ancilla + num_qubits, shots=shots)

    # Remap Hamiltonian to match wire order in QPE
    H_sys = hamiltonian.map_wires(dict(zip(range(num_qubits), system_wires)))

    @qml.qnode(dev)
    def circuit():
        # Prepare Hartree–Fock state on system register
        qml.BasisState(np.array(hf_state, dtype=int), wires=system_wires)

        # Apply Hadamards on ancilla qubits
        for a in ancilla_wires:
            qml.Hadamard(wires=a)

        # Controlled U^(2^k)
        for k, a in enumerate(ancilla_wires):
            n_repeat = 2 ** (n_ancilla - 1 - k)
            for _ in range(n_repeat):
                qml.ctrl(qml.ApproxTimeEvolution, control=a)(
                    H_sys, t, trotter_steps
                )
                if noise_params:
                    apply_noise_all(
                        wires=system_wires + [a],
                        p_dep=noise_params.get("p_dep", 0.0),
                        p_amp=noise_params.get("p_amp", 0.0),
                    )

        # Apply inverse QFT on ancilla register
        inverse_qft(ancilla_wires)

        return qml.sample(wires=ancilla_wires)

    # Execute circuit
    samples = np.array(circuit(), dtype=int)

    # Convert results to bitstrings
    bitstrings = ["".join(str(int(b)) for b in s) for s in samples]

    counts = {}
    for b in bitstrings:
        counts[b] = counts.get(b, 0) + 1
    probs = {b: c / shots for b, c in counts.items()}

    # Compute Hartree–Fock energy (reference)
    E_hf = hartree_fock_energy(
        H_sys.map_wires(dict(zip(system_wires, range(num_qubits)))), hf_state
    )

    # Build records with phase + energy
    rows = []
    for b, c in counts.items():
        ph_m = bitstring_to_phase(b, msb_first=True)
        ph_l = bitstring_to_phase(b, msb_first=False)
        e_m = phase_to_energy_unwrapped(ph_m, t, ref_energy=E_hf)
        e_l = phase_to_energy_unwrapped(ph_l, t, ref_energy=E_hf)
        rows.append((b, c, ph_m, ph_l, e_m, e_l))

    # Choose best estimate by frequency
    best_row = max(rows, key=lambda r: r[1])
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


# ---------------------------------------------------------------------
# Hartree–Fock Reference Energy
# ---------------------------------------------------------------------
def hartree_fock_energy(hamiltonian, hf_state):
    """Compute ⟨HF|H|HF⟩ as a reference energy in Hartree."""
    num_qubits = len(hf_state)
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(hf_state, wires=range(num_qubits))
        return qml.expval(hamiltonian)

    return float(circuit())


# ---------------------------------------------------------------------
# Phase / Energy Conversions
# ---------------------------------------------------------------------
def bitstring_to_phase(bits: str, msb_first: bool = True) -> float:
    """Convert a 0/1 string to fractional phase ∈ [0,1)."""
    b = bits if msb_first else bits[::-1]
    frac = 0.0
    for i, ch in enumerate(b, start=1):
        frac += (ch == "1") * (0.5 ** i)
    return float(frac)


def phase_to_energy_unwrapped(
    phase: float, t: float, ref_energy: Optional[float] = None
) -> float:
    """Convert phase → energy, unwrap modulo 2π, and match reference energy branch."""
    base = 2 * np.pi * phase / t
    energy = -base

    # Wrap into (-π/t, π/t]
    if energy > np.pi / t:
        energy -= 2 * np.pi / t
    elif energy <= -np.pi / t:
        energy += 2 * np.pi / t

    # Adjust sign branch near ref_energy
    if ref_energy is not None:
        candidates = [energy + k * (2 * np.pi / t) for k in (-1, 0, 1)]
        energy = min(candidates, key=lambda x: abs(x - ref_energy))

    return float(energy)
