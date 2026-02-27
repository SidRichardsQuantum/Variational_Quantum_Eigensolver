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
    *,
    hamiltonian: qml.Hamiltonian,
    hf_state: np.ndarray,
    seed: int = 0,
    n_ancilla: int = 4,
    t: float = 1.0,
    trotter_steps: int = 1,
    noise_params: Optional[Dict[str, float]] = None,
    shots: Optional[int] = 1000,
    molecule_name: str = "molecule",
    system_qubits: Optional[int] = None,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    force: bool = False,
) -> Dict[str, Any]:
    """
    Run a (noisy or noiseless) Quantum Phase Estimation simulation with caching.

    Caching
    -------
    - force=False (default): return cached result if available; otherwise run and save.
    - force=True: ignore cache, run and overwrite/save.

    Notes
    -----
    Cache keys and filenames are managed by qpe.io_utils. In particular, qpe.io_utils
    uses `noise or {}` in its signature hash, so we normalise "no noise" to {}.
    """
    # Local import to keep qpe.core usable without I/O side effects at import time
    from qpe.io_utils import (
        ensure_dirs,
        load_qpe_result,
        save_qpe_result,
    )

    ensure_dirs()

    # -------------------------
    # Normalise inputs for stable caching
    # -------------------------
    # 1) Shots are part of the cache key; keep stable typing
    shots_i = int(shots)

    # 2) Noise: collapse None, {}, and {"p_dep":0,"p_amp":0} to "no noise"
    norm_noise = canonical_noise(
        noisy=True,
        p_dep=float((noise_params or {}).get("p_dep", 0.0)),
        p_amp=float((noise_params or {}).get("p_amp", 0.0)),
        model=None,
    )

    # -------------------------
    # Cache lookup
    # -------------------------
    if not force:
        cached = load_qpe_result(
            molecule=molecule_name,
            n_ancilla=int(n_ancilla),
            t=float(t),
            seed=int(seed),
            shots=shots_i,
            noise=(norm_noise or None),  # io_utils will treat None as {}
            trotter_steps=int(trotter_steps),
            mapping=str(mapping),
            unit=str(unit),
        )
        if cached is not None:
            return cached

    # -------------------------
    # Compute
    # -------------------------
    num_qubits = len(hf_state)
    np.random.seed(int(seed))

    ancilla_wires = list(range(n_ancilla))
    system_wires = list(range(n_ancilla, n_ancilla + num_qubits))

    dev_name = "default.mixed" if bool(norm_noise) else "default.qubit"
    dev = qml.device(dev_name, wires=n_ancilla + num_qubits)

    # Remap Hamiltonian wires to system register indices
    wire_map = {i: system_wires[i] for i in range(num_qubits)}
    H_sys = hamiltonian.map_wires(wire_map)

    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array(hf_state, dtype=int), wires=system_wires)

        for a in ancilla_wires:
            qml.Hadamard(wires=a)

        for k, a in enumerate(ancilla_wires):
            power = n_ancilla - 1 - k
            controlled_powered_evolution(
                hamiltonian=H_sys,
                system_wires=system_wires,
                control_wire=a,
                t=t,
                power=power,
                trotter_steps=trotter_steps,
                noise_params=(norm_noise or None),
            )

        inverse_qft(ancilla_wires)
        return qml.sample(wires=ancilla_wires)

    if shots_i is not None:
        circuit = qml.set_shots(shots_i)(circuit)

    samples = np.array(circuit(), dtype=int)
    samples = np.atleast_2d(samples)

    bitstrings = ["".join(str(int(b)) for b in s) for s in samples]
    counts = dict(Counter(bitstrings))
    probs = {b: c / shots_i for b, c in counts.items()}

    E_hf = hartree_fock_energy(hamiltonian, hf_state)

    rows = []
    for b, c in counts.items():
        ph_m = bitstring_to_phase(b, msb_first=True)
        ph_l = bitstring_to_phase(b, msb_first=False)
        e_m = phase_to_energy_unwrapped(ph_m, t, ref_energy=E_hf)
        e_l = phase_to_energy_unwrapped(ph_l, t, ref_energy=E_hf)
        rows.append((b, c, ph_m, ph_l, e_m, e_l))

    if not rows:
        raise RuntimeError("QPE returned no measurement outcomes.")

    best_row = max(rows, key=lambda r: r[1])
    best_b = best_row[0]

    candidate_Es = (best_row[4], best_row[5])
    best_energy = min(candidate_Es, key=lambda x: abs(x - E_hf))
    best_phase = best_row[2] if best_energy == best_row[4] else best_row[3]

    result: Dict[str, Any] = {
        "molecule": molecule_name,
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
        "mapping": str(mapping),
        "unit": str(unit),
    }

    if system_qubits is not None:
        result["system_qubits"] = int(system_qubits)

    save_qpe_result(result)
    return result
