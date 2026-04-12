"""
qpe.core
========
Core Quantum Phase Estimation (QPE) implementation.

This module is deliberately focused on:
    • Circuit construction (QPE, with optional noise)
    • Classical post-processing (bitstrings → phases → energies)
    • A thin caching layer for "run" entrypoints, so notebooks/CLI stay clean

It does **not**:
    • Build Hamiltonians (see qpe.hamiltonian)
    • Plot (see qpe.visualize)
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional

import pennylane as qml
from pennylane import numpy as np

from common.persist import canonical_noise
from common.problem import resolve_problem
from qpe.noise import apply_noise_all


# ---------------------------------------------------------------------
# Inverse Quantum Fourier Transform
# ---------------------------------------------------------------------
def inverse_qft(wires: list[int]) -> None:
    """
    Apply the inverse Quantum Fourier Transform (QFT) on a list of wires.

    The input is assumed to be ordered [a_0, a_1, ..., a_{n-1}]
    with a_0 the most-significant ancilla.
    """
    n = len(wires)

    # Mirror ordering
    for i in range(n // 2):
        qml.SWAP(wires=[wires[i], wires[n - i - 1]])

    # Controlled phase ladder + Hadamards
    for j in range(n):
        k = n - j - 1
        qml.Hadamard(wires=k)
        for m in range(k):
            angle = -np.pi / (2 ** (k - m))
            qml.ControlledPhaseShift(angle, wires=[wires[m], wires[k]])


# ---------------------------------------------------------------------
# Controlled powered evolution U^(2^power)
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
    """
    Apply controlled-U^(2^power) = controlled exp(-i H t 2^power).

    Uses ApproxTimeEvolution in PennyLane, with optional noise applied
    after each controlled segment.

    Args
    ----
    hamiltonian:
        Molecular Hamiltonian acting on the *system* wires.
        (Already mapped onto system_wires.)
    system_wires:
        Wires of the system register.
    control_wire:
        Ancilla controlling the evolution.
    t:
        Base evolution time in exp(-i H t).
    power:
        Exponent; this block implements U^(2^power).
    trotter_steps:
        Number of Trotter steps per exp(-i H t).
    noise_params:
        Optional dict {"p_dep": float, "p_amp": float}.
    """
    n_repeat = 2**power

    for _ in range(n_repeat):
        # Controlled ApproxTimeEvolution
        qml.ctrl(qml.ApproxTimeEvolution, control=control_wire)(
            hamiltonian, t, trotter_steps
        )

        # Noise on all active wires
        if noise_params:
            apply_noise_all(
                wires=system_wires + [control_wire],
                p_dep=noise_params.get("p_dep", 0.0),
                p_amp=noise_params.get("p_amp", 0.0),
            )


# ---------------------------------------------------------------------
# Hartree–Fock Reference Energy
# ---------------------------------------------------------------------
def hartree_fock_energy(hamiltonian: qml.Hamiltonian, hf_state: np.ndarray) -> float:
    """Compute ⟨HF|H|HF⟩ in Hartree."""
    num_qubits = len(hf_state)
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(hf_state, wires=range(num_qubits))
        return qml.expval(hamiltonian)

    return float(circuit())


# ---------------------------------------------------------------------
# Phase / Energy utilities
# ---------------------------------------------------------------------
def bitstring_to_phase(bits: str, msb_first: bool = True) -> float:
    """
    Convert bitstring → fractional phase in [0, 1).

    Args
    ----
    bits:
        String of "0"/"1", e.g. "0110".
    msb_first:
        If False, interpret the string as LSB-first.

    Returns
    -------
    float
        Phase in [0, 1).
    """
    b = bits if msb_first else bits[::-1]
    return float(sum((ch == "1") * (0.5**i) for i, ch in enumerate(b, start=1)))


def phase_to_energy_unwrapped(
    phase: float,
    t: float,
    ref_energy: Optional[float] = None,
) -> float:
    """
    Convert a phase in [0, 1) into an energy, unwrapped around a reference.

    The base relation is:
        E ≈ -2π * phase / t   (mod 2π / t)

    We first wrap E into (-π/t, π/t], then (if ref_energy is given) shift
    by ± 2π/t to choose the branch closest to ref_energy.
    """
    base = -2 * np.pi * phase / t

    # Wrap into (-π/t, π/t]
    while base > np.pi / t:
        base -= 2 * np.pi / t
    while base <= -np.pi / t:
        base += 2 * np.pi / t

    if ref_energy is not None:
        spaced = 2 * np.pi / t
        candidates = [base + k * spaced for k in (-1, 0, 1)]
        base = min(candidates, key=lambda x: abs(x - ref_energy))

    return float(base)


# ---------------------------------------------------------------------
# QPE main runner (cached)
# ---------------------------------------------------------------------
def run_qpe(
    molecule: str = "H2",
    seed: int = 0,
    n_ancilla: int = 4,
    t: float = 1.0,
    trotter_steps: int = 1,
    shots: int | None = 1000,
    plot: bool = True,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    phase_damping_prob: float = 0.0,
    bit_flip_prob: float = 0.0,
    phase_flip_prob: float = 0.0,
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    charge: int = 0,
    unit: str = "angstrom",
    mapping: str = "jordan_wigner",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    hamiltonian: qml.Hamiltonian | None = None,
    hf_state: np.ndarray | None = None,
    system_qubits: int | None = None,
) -> Dict[str, Any]:
    """
    Run a (noisy or noiseless) Quantum Phase Estimation simulation with caching.

    High-level API mirrors run_vqe():
    - registry mode via `molecule=...`
    - explicit geometry mode via `symbols=..., coordinates=..., charge=..., basis=...`
    - optional expert override via precomputed `hamiltonian` and `hf_state`
    """
    # Local import to keep qpe.core usable without I/O side effects at import time
    from qpe.io_utils import ensure_dirs, load_qpe_result, save_qpe_result

    ensure_dirs()
    np.random.seed(int(seed))
    if (hamiltonian is None) != (hf_state is None):
        raise ValueError(
            "QPE expert mode requires both hamiltonian and hf_state together."
        )

    problem = resolve_problem(
        molecule=molecule,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        charge=charge,
        mapping=mapping,
        unit=unit,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian=hamiltonian,
        num_qubits=system_qubits,
        reference_state=hf_state,
        require_reference_state=hamiltonian is not None,
        reference_name="hf_state",
    )
    H = problem.hamiltonian
    hf_bits = np.array(problem.reference_state, dtype=int)
    molecule_label = problem.molecule_label
    symbols_out = problem.symbols
    coordinates_out = problem.coordinates
    basis_out = problem.basis
    charge_out = problem.charge
    unit_out = problem.unit
    qubits = problem.num_qubits
    resolved_active_electrons = problem.active_electrons
    resolved_active_orbitals = problem.active_orbitals
    mapping_norm = problem.mapping
    cache_enabled = problem.cacheable

    # -------------------------
    # Normalise noise
    # -------------------------
    norm_noise = canonical_noise(
        noisy=bool(noisy),
        p_dep=float(depolarizing_prob),
        p_amp=float(amplitude_damping_prob),
        p_phase_damp=float(phase_damping_prob),
        p_bit_flip=float(bit_flip_prob),
        p_phase_flip=float(phase_flip_prob),
        model=None,
    )

    shots_i = None if shots is None else int(shots)

    # -------------------------
    # Cache lookup
    # -------------------------
    if cache_enabled and not force:
        cached = load_qpe_result(
            molecule=molecule_label,
            symbols=list(symbols_out),
            geometry=np.array(coordinates_out, dtype=float),
            basis=str(basis_out),
            charge=int(charge_out),
            n_ancilla=int(n_ancilla),
            t=float(t),
            seed=int(seed),
            shots=shots_i,
            noise=(norm_noise or None),
            trotter_steps=int(trotter_steps),
            mapping=mapping_norm,
            unit=unit_out,
            active_electrons=resolved_active_electrons,
            active_orbitals=resolved_active_orbitals,
        )
        if cached is not None:
            return cached

    # -------------------------
    # Compute
    # -------------------------
    num_qubits = len(hf_bits)

    ancilla_wires = list(range(int(n_ancilla)))
    system_wires = list(range(int(n_ancilla), int(n_ancilla) + num_qubits))

    dev_name = "default.mixed" if bool(norm_noise) else "default.qubit"
    dev = qml.device(dev_name, wires=int(n_ancilla) + num_qubits)

    wire_map = {i: system_wires[i] for i in range(num_qubits)}
    H_sys = H.map_wires(wire_map)

    analytic_mode = shots_i is None

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array(hf_bits, dtype=int), wires=system_wires)

        for a in ancilla_wires:
            qml.Hadamard(wires=a)

        for k, a in enumerate(ancilla_wires):
            power = int(n_ancilla) - 1 - k
            controlled_powered_evolution(
                hamiltonian=H_sys,
                system_wires=system_wires,
                control_wire=a,
                t=float(t),
                power=power,
                trotter_steps=int(trotter_steps),
                noise_params=(norm_noise or None),
            )

        inverse_qft(ancilla_wires)
        if analytic_mode:
            return qml.probs(wires=ancilla_wires)
        return qml.sample(wires=ancilla_wires)

    if shots_i is not None:
        circuit = qml.set_shots(circuit, shots=shots_i)

    if analytic_mode:
        prob_vector = np.array(circuit(), dtype=float).reshape(-1)
        probs = {
            format(idx, f"0{int(n_ancilla)}b"): float(prob)
            for idx, prob in enumerate(prob_vector)
            if float(prob) > 0.0
        }
        counts = dict(probs)
    else:
        samples = np.array(circuit(), dtype=int)
        samples = np.atleast_2d(samples)

        bitstrings = ["".join(str(int(b)) for b in s) for s in samples]
        counts = dict(Counter(bitstrings))
        probs = {b: c / len(bitstrings) for b, c in counts.items()}

    E_hf = hartree_fock_energy(H, hf_bits)

    rows = []
    for b, weight in probs.items():
        ph_m = bitstring_to_phase(b, msb_first=True)
        ph_l = bitstring_to_phase(b, msb_first=False)
        e_m = phase_to_energy_unwrapped(ph_m, float(t), ref_energy=E_hf)
        e_l = phase_to_energy_unwrapped(ph_l, float(t), ref_energy=E_hf)
        rows.append((b, float(weight), ph_m, ph_l, e_m, e_l))

    if not rows:
        raise RuntimeError("QPE returned no measurement outcomes.")

    best_row = max(rows, key=lambda r: r[1])
    best_b = best_row[0]

    candidate_Es = (best_row[4], best_row[5])
    best_energy = min(candidate_Es, key=lambda x: abs(x - E_hf))
    best_phase = best_row[2] if best_energy == best_row[4] else best_row[3]

    result: Dict[str, Any] = {
        "molecule": molecule_label,
        "symbols": list(symbols_out),
        "geometry": np.array(coordinates_out, dtype=float).tolist(),
        "basis": basis_out,
        "charge": charge_out,
        "unit": unit_out,
        "counts": counts,
        "probs": probs,
        "best_bitstring": best_b,
        "phase": float(best_phase),
        "energy": float(best_energy),
        "hf_energy": float(E_hf),
        "n_ancilla": int(n_ancilla),
        "trotter_steps": int(trotter_steps),
        "t": float(t),
        "seed": int(seed),
        "noise": dict(norm_noise),
        "shots": shots_i,
        "mapping": mapping_norm,
        "active_electrons": resolved_active_electrons,
        "active_orbitals": resolved_active_orbitals,
        "num_qubits": int(qubits),
    }

    if plot:
        from qpe.visualize import plot_qpe_distribution

        plot_qpe_distribution(result, show=False, save=True)

    if cache_enabled:
        save_qpe_result(result)
    return result
