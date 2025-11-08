"""
Command-line interface for QPE simulations.

Usage examples:
    python -m qpe --molecule H2
    python -m qpe --molecule H2 --noisy --p_dep 0.05 --p_amp 0.02
    python -m qpe --molecule H2 --ancillas 6 --t 1.0 --shots 2000
"""

import argparse
import json
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
    apply_noise_all,
)


# ---------------------------------------------------------------------
# Molecule definitions
# ---------------------------------------------------------------------
MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "coordinates": np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.7414]]),  # Å
        "charge": 0,
        "basis": "STO-3G",
        "electrons": 2,
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "coordinates": np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.6]]),  # Å
        "charge": 0,
        "basis": "STO-3G",
        "electrons": 4,
    },
    "H3+": {
        "symbols": ["H", "H", "H"],
        "coordinates": np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.85],
            [0.0, 0.0, 1.70]
        ]),
        "charge": +1,
        "basis": "STO-3G",
        "electrons": 2,
    },
}


# ---------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="QPE simulation runner."
    )

    parser.add_argument(
        "--molecule", type=str, choices=MOLECULES.keys(), required=True,
        help="Which molecule to simulate (H2, LiH, H3+)"
    )
    parser.add_argument("--ancillas", type=int, default=4,
                        help="Number of ancilla qubits (default=4)")
    parser.add_argument("--t", type=float, default=1.0,
                        help="Evolution time in exp(-iHt) (default=1.0)")
    parser.add_argument("--trotter", type=int, default=2,
                        help="Trotter steps for time evolution (default=2)")
    parser.add_argument("--shots", type=int, default=1000,
                        help="Number of measurement shots (default=1000)")

    # Noise flags
    parser.add_argument("--noisy", action="store_true",
                        help="Enable noise model (depolarizing / amplitude damping)")
    parser.add_argument("--p_dep", type=float, default=0.0,
                        help="Depolarizing noise probability (default=0.0)")
    parser.add_argument("--p_amp", type=float, default=0.0,
                        help="Amplitude damping probability (default=0.0)")

    parser.add_argument("--save-plot", action="store_true",
                        help="Save QPE distribution plot")
    parser.add_argument("--no-plot", action="store_true",
                        help="Run without plotting output")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dirs()

    mol = MOLECULES[args.molecule]
    print(f"🔹 Running QPE for {args.molecule} ({mol['basis']})")

    # Build Hamiltonian
    H, num_qubits = qchem.molecular_hamiltonian(
        mol["symbols"], mol["coordinates"],
        charge=mol["charge"], basis=mol["basis"]
    )
    hf_state = qchem.hf_state(mol["electrons"], num_qubits)

    noise_params = None
    if args.noisy and (args.p_dep > 0 or args.p_amp > 0):
        noise_params = {"p_dep": args.p_dep, "p_amp": args.p_amp}
        print(f"🌀 Using noise model: dep={args.p_dep}, amp={args.p_amp}")

    # Cache signature
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
            trotter_steps=args.trotter,
            noise_params=noise_params,
            shots=args.shots,
            molecule_name=args.molecule,
        )
        save_qpe_result(result)

    # Display summary
    print("\n✅ QPE completed.")
    print(f"Most probable state: {result['best_bitstring']}")
    print(f"Estimated phase: {result['phase']:.6f}")
    print(f"Estimated energy: {result['energy']:.8f} Ha")

    E_hf = hartree_fock_energy(H, hf_state)
    print(f"Hartree–Fock energy: {E_hf:.8f} Ha")
    print(f"ΔE (QPE - HF): {result['energy'] - E_hf:+.8f} Ha")

    # Plot
    if not args.no_plot:
        plot_qpe_distribution(result, show=True, save=args.save_plot)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
