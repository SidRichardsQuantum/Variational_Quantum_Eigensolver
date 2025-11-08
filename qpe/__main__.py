"""
Command-line interface for Quantum Phase Estimation (QPE) simulations.

Examples:
    python -m qpe --molecule H2
    python -m qpe --molecule H3+ --ancillas 5 --t 0.8 --save-plot
    python -m qpe --molecule LiH --noisy --p_dep 0.05 --shots 2000
"""

import argparse
import time
from typing import Dict, Any

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

from qpe import (
    run_qpe,
    save_qpe_result,
    load_qpe_result,
    plot_qpe_distribution,
    hartree_fock_energy,
    signature_hash,
    ensure_dirs,
)

# -----------------------------
# Molecule configurations
# -----------------------------
# Charge included; electrons will be inferred from symbols & charge.
MOLECULES: Dict[str, Dict[str, Any]] = {
    "H2": {
        "symbols": ["H", "H"],
        "coordinates": np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.7414]]),  # Å
        "charge": 0,
        "basis": "STO-3G",
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "coordinates": np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.6]]),  # Å
        "charge": 0,
        "basis": "STO-3G",
    },
    "H2O": {
        "symbols": ["O", "H", "H"],
        "coordinates": np.array([[0.000000, 0.000000, 0.000000],
                                 [0.758602, 0.000000, 0.504284],
                                 [-0.758602, 0.000000, 0.504284]]),  # Å
        "charge": 0,
        "basis": "STO-3G",
    },
    "H3+": {
        "symbols": ["H", "H", "H"],
        # Mildly non-equilateral example geometry
        "coordinates": np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.872],
                                 [0.755, 0.0, 0.436]]),  # Å
        "charge": +1,
        "basis": "STO-3G",
    },
}

# Minimal Z table for electron counting (extend if you add more elements)
Z = {"H": 1, "Li": 3, "O": 8}


def infer_electrons(symbols, charge: int) -> int:
    return int(sum(Z[s] for s in symbols) - charge)


def build_hamiltonian_safe(symbols, coordinates, charge, basis):
    """
    Build molecular Hamiltonian; fall back to OpenFermion backend for open-shell cases.
    """
    try:
        # First try default backend
        H, n_qubits = qchem.molecular_hamiltonian(
            symbols, coordinates, charge=charge, basis=basis
        )
        return H, n_qubits
    except Exception as e_primary:
        # Fallback to OpenFermion backend (supports open-shell)
        try:
            H, n_qubits = qchem.molecular_hamiltonian(
                symbols, coordinates, charge=charge, basis=basis, method="openfermion"
            )
            return H, n_qubits
        except Exception as e_fallback:
            msg = (
                f"Failed to build Hamiltonian with default and OpenFermion backends.\n"
                f"Default backend error: {e_primary}\n"
                f"OpenFermion error: {e_fallback}\n"
                f"If the fallback complains about missing dependencies, install:\n"
                f"    pip install openfermion openfermionpyscf\n"
            )
            raise RuntimeError(msg)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Phase Estimation (QPE) simulation runner."
    )
    parser.add_argument(
        "--molecule", type=str, choices=MOLECULES.keys(), required=True,
        help="Molecule to simulate (H2, H3+, LiH, H2O)"
    )
    parser.add_argument("--ancillas", type=int, default=4,
                        help="Number of ancilla qubits (default: 4)")
    parser.add_argument("--t", type=float, default=1.0,
                        help="Evolution time in exp(-i H t) (default: 1.0)")
    parser.add_argument("--trotter-steps", type=int, default=2,
                        help="Trotter steps for time evolution (default: 2)")
    parser.add_argument("--shots", type=int, default=1000,
                        help="Number of measurement shots (default: 1000)")

    # Noise
    parser.add_argument("--noisy", action="store_true",
                        help="Enable noise model (depolarizing / amplitude damping)")
    parser.add_argument("--p_dep", type=float, default=0.0,
                        help="Depolarizing probability (default: 0.0)")
    parser.add_argument("--p_amp", type=float, default=0.0,
                        help="Amplitude damping probability (default: 0.0)")

    # Plotting
    parser.add_argument("--save-plot", action="store_true",
                        help="Save QPE distribution plot to qpe/images/")
    parser.add_argument("--no-plot", action="store_true",
                        help="Do not display the plot")

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dirs()

    cfg = MOLECULES[args.molecule]
    symbols = cfg["symbols"]
    coords = cfg["coordinates"]
    charge = int(cfg["charge"])
    basis = cfg["basis"]

    electrons = infer_electrons(symbols, charge)

    print(f"🔹 Running QPE for {args.molecule} ({basis})")
    t0 = time.time()

    # Hamiltonian build with robust fallback
    H, n_qubits = build_hamiltonian_safe(symbols, coords, charge, basis)

    # HF state on returned spin-orbital space
    hf_state = qchem.hf_state(electrons, n_qubits)

    total_qubits = n_qubits + args.ancillas
    if total_qubits >= 20:
        print(f"⚠️  Warning: total qubits = system({n_qubits}) + ancillas({args.ancillas}) = {total_qubits}. "
              f"Simulations may be slow.")

    noise_params = None
    if args.noisy and (args.p_dep > 0.0 or args.p_amp > 0.0):
        noise_params = {"p_dep": float(args.p_dep), "p_amp": float(args.p_amp)}
        print(f"🌀 Noise enabled: dep={args.p_dep}, amp={args.p_amp}")

    # Cache key
    sig = signature_hash(
        molecule=args.molecule,
        n_ancilla=args.ancillas,
        t=args.t,
        noise=noise_params,
        shots=args.shots,
    )

    cached = load_qpe_result(args.molecule, sig)
    if cached:
        print("✅ Loaded cached QPE result.")
        result = cached
    else:
        print("▶️ Running QPE simulation...")
        result = run_qpe(
            hamiltonian=H,
            hf_state=hf_state,
            n_ancilla=args.ancillas,
            t=args.t,
            trotter_steps=args.trotter_steps,
            noise_params=noise_params,
            shots=args.shots,
            molecule_name=args.molecule,
        )
        save_qpe_result(result)

    dt = time.time() - t0

    # Summary
    print("\n✅ QPE completed.")
    print(f"Most probable state: {result['best_bitstring']}")
    print(f"Estimated phase: {result['phase']:.6f}")
    print(f"Estimated energy: {result['energy']:.8f} Ha")

    E_hf = hartree_fock_energy(H, hf_state)
    print(f"Hartree–Fock energy: {E_hf:.8f} Ha")
    print(f"ΔE (QPE - HF): {result['energy'] - E_hf:+.8f} Ha")
    print(f"⏱  Elapsed: {dt:.2f}s | Qubits: system={n_qubits}, ancillas={args.ancillas} (total {n_qubits + args.ancillas})")

    # Plot
    if not args.no_plot:
        plot_qpe_distribution(result, show=True, save=args.save_plot)


if __name__ == "__main__":
    main()
