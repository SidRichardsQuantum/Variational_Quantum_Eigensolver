"""
qpe/core.py
===========
Core Quantum Phase Estimation (QPE) implementation.

Supports both noiseless and noisy QPE simulations using PennyLane.
Integrates cleanly with qpe.io_utils and qpe.noise modules.
"""

from __future__ import annotations
import pennylane as qml
from pennylane import numpy as np
from typing import Dict, Any, Optional
from collections import Counter

from qpe.noise import apply_noise_all
from qpe.io_utils import save_qpe_result


# ---------------------------------------------------------------------
# Inverse Quantum Fourier Transform
# ---------------------------------------------------------------------
def inverse_qft(wires: list[int]) -> None:
    """Apply the inverse Quantum Fourier Transform (QFT) on given wires."""
    n = len(wires)
    # Swap order (mirror)
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - i - 1]])

    # Apply Hadamard + controlled phase rotations
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
    hamiltonian: qml.Hamiltonian,
    system_wires: list[int],
    control_wire: int,
    t: float,
    power: int,
    trotter_steps: int = 1,
    noise_params: Optional[Dict[str, float]] = None,
) -> None:
    """Apply controlled-U^(2^power) = controlled exp(-i * H * t * 2^power).

    Args:
        hamiltonian: PennyLane Hamiltonian.
        system_wires: Target system wires.
        control_wire: Single ancilla wire controlling evolution.
        t: Evolution time per step.
        power: Exponent controlling repetition count.
        trotter_steps: Number of Trotter steps for ApproxTimeEvolution.
        noise_params: Optional dict with keys {'p_dep', 'p_amp'}.
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
    hamiltonian: qml.Hamiltonian,
    hf_state: np.ndarray,
    n_ancilla: int = 4,
    t: float = 1.0,
    trotter_steps: int = 1,
    noise_params: Optional[Dict[str, float]] = None,
    shots: int = 5000,
    molecule_name: str = "molecule",
    save: bool = True,
) -> Dict[str, Any]:
    """Run a (noisy or noiseless) Quantum Phase Estimation simulation.

    Args:
        hamiltonian: Molecular Hamiltonian operator.
        hf_state: Binary Hartree–Fock state vector.
        n_ancilla: Number of ancilla (phase) qubits.
        t: Time evolution parameter in exp(-iHt).
        trotter_steps: Number of Trotter steps for ApproxTimeEvolution.
        noise_params: Optional dict with {'p_dep', 'p_amp'}.
        shots: Number of measurement samples.
        molecule_name: Label for saving and logging.
        save: Whether to cache result JSON in `package_results/`.

    Returns:
        dict with:
            counts, probs, phase, energy, best_bitstring,
            hf_energy, n_ancilla, t, noise, shots, molecule
    """
    num_qubits = len(hf_state)
    ancilla_wires = list(range(n_ancilla))
    system_wires = list(range(n_ancilla, n_ancilla + num_qubits))

    # Device selection
    dev_name = "default.mixed" if noise_params else "default.qubit"
    dev = qml.device(dev_name, wires=n_ancilla + num_qubits, shots=shots)

    # Remap Hamiltonian wires to match system layout
    H_sys = hamiltonian.map_wires({i: system_wires[i] for i in range(num_qubits)})

    @qml.qnode(dev)
    def circuit():
        # Prepare HF state
        qml.BasisState(np.array(hf_state, dtype=int), wires=system_wires)

        # Apply Hadamards on ancilla register
        for a in ancilla_wires:
            qml.Hadamard(wires=a)

        # Controlled unitary evolution
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

        # Inverse QFT
        inverse_qft(ancilla_wires)
        return qml.sample(wires=ancilla_wires)

    # Run circuit
    samples = np.array(circuit(), dtype=int)
    bitstrings = ["".join(str(int(b)) for b in s) for s in samples]
    counts = dict(Counter(bitstrings))
    probs = {b: c / shots for b, c in counts.items()}

    # Compute HF reference
    E_hf = hartree_fock_energy(hamiltonian, hf_state)

    # Phase / energy reconstruction
    rows = []
    for b, c in counts.items():
        ph_m = bitstring_to_phase(b, msb_first=True)
        ph_l = bitstring_to_phase(b, msb_first=False)
        e_m = phase_to_energy_unwrapped(ph_m, t, ref_energy=E_hf)
        e_l = phase_to_energy_unwrapped(ph_l, t, ref_energy=E_hf)
        rows.append((b, c, ph_m, ph_l, e_m, e_l))

    best_row = max(rows, key=lambda r: r[1])
    best_b = best_row[0]
    best_E = min((best_row[4], best_row[5]), key=lambda x: abs(x - E_hf))
    best_phase = best_row[2] if best_E == best_row[4] else best_row[3]

    result = {
        "molecule": molecule_name,
        "counts": counts,
        "probs": probs,
        "best_bitstring": best_b,
        "phase": float(best_phase),
        "energy": float(best_E),
        "hf_energy": float(E_hf),
        "n_ancilla": n_ancilla,
        "t": t,
        "noise": noise_params or {},
        "shots": shots,
    }

    if save:
        save_qpe_result(result)

    return result


# ---------------------------------------------------------------------
# Hartree–Fock Reference Energy
# ---------------------------------------------------------------------
def hartree_fock_energy(hamiltonian: qml.Hamiltonian, hf_state: np.ndarray) -> float:
    """Compute ⟨HF|H|HF⟩ energy in Hartree."""
    num_qubits = len(hf_state)
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(hf_state, wires=range(num_qubits))
        return qml.expval(hamiltonian)

    return float(circuit())


# ---------------------------------------------------------------------
# Phase / Energy Conversion Utilities
# ---------------------------------------------------------------------
def bitstring_to_phase(bits: str, msb_first: bool = True) -> float:
    """Convert a bitstring (MSB or LSB first) into a fractional phase in [0, 1)."""
    b = bits if msb_first else bits[::-1]
    return float(sum((ch == "1") * (0.5 ** i) for i, ch in enumerate(b, start=1)))


def phase_to_energy_unwrapped(
    phase: float, t: float, ref_energy: Optional[float] = None
) -> float:
    """Convert phase → energy, unwrapping modulo 2π around reference."""
    base = -2 * np.pi * phase / t

    # Wrap into (-π/t, π/t]
    while base > np.pi / t:
        base -= 2 * np.pi / t
    while base <= -np.pi / t:
        base += 2 * np.pi / t

    if ref_energy is not None:
        candidates = [base + k * (2 * np.pi / t) for k in (-1, 0, 1)]
        base = min(candidates, key=lambda x: abs(x - ref_energy))

    return float(base)
