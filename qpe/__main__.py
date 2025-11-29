"""
qpe.__main__
============
Command-line interface for Quantum Phase Estimation (QPE).

This CLI mirrors the modern structure of the VQE CLI:
    • clean argument parsing
    • shared plotting conventions (common.plotting)
    • cached result loading
    • separation of concerns (no logic mixed with plotting or circuit code)

Example:
    python -m qpe --molecule H2 --ancillas 4 --t 1.0 --shots 2000
"""

from __future__ import annotations
import argparse
import time

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

from qpe.hamiltonian import build_hamiltonian
from qpe.core import run_qpe
from qpe.io_utils import (
    save_qpe_result,
    load_qpe_result,
    signature_hash,
    ensure_dirs,
)
from qpe.visualize import plot_qpe_distribution


# ---------------------------------------------------------------------
# Molecule presets (simple & script-friendly)
# ---------------------------------------------------------------------
MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "coordinates": np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 0.7414]]),
        "charge": 0,
        "basis": "STO-3G",
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "coordinates": np.array([[0.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.6]]),
        "charge": 0,
        "basis": "STO-3G",
    },
    "H2O": {
        "symbols": ["O", "H", "H"],
        "coordinates": np.array([
            [0.000000, 0.000000, 0.000000],
            [0.758602, 0.000000, 0.504284],
            [-0.758602, 0.000000, 0.504284],
        ]),
        "charge": 0,
        "basis": "STO-3G",
    },
    "H3+": {
        "symbols": ["H", "H", "H"],
        "coordinates": np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.872],
            [0.755, 0.0, 0.436],
        ]),
        "charge": +1,
        "basis": "STO-3G",
    },
}

# Minimal atomic number table for electron counting
Z = {"H": 1, "Li": 3, "O": 8}


def infer_electrons(symbols, charge: int) -> int:
    """Infer number of electrons for a molecule from symbols and charge."""
    return int(sum(Z[s] for s in symbols) - charge)


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Phase Estimation (QPE) simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--molecule",
        required=True,
        choices=MOLECULES.keys(),
        help="Molecule to simulate",
    )

    parser.add_argument(
        "--ancillas", type=int, default=4,
        help="Number of ancilla qubits",
    )

    parser.add_argument(
        "--t", type=float, default=1.0,
        help="Evolution time in exp(-iHt)",
    )

    parser.add_argument(
        "--trotter-steps", type=int, default=2,
        help="Trotter steps for time evolution",
    )

    parser.add_argument(
        "--shots", type=int, default=2000,
        help="Number of measurement shots",
    )

    # Noise model
    parser.add_argument("--noisy", action="store_true",
                        help="Enable noise model")
    parser.add_argument("--p-dep", type=float, default=0.0,
                        help="Depolarizing probability")
    parser.add_argument("--p-amp", type=float, default=0.0,
                        help="Amplitude damping probability")

    # Plotting
    parser.add_argument("--plot", action="store_true",
                        help="Show plot after simulation")
    parser.add_argument("--save-plot", action="store_true",
                        help="Save QPE probability distribution")

    parser.add_argument("--force", action="store_true",
                        help="Force rerun even if cached result exists")

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dirs()

    cfg = MOLECULES[args.molecule]

    print(f"\n🧮  QPE Simulation")
    print(f"• Molecule:   {args.molecule}")
    print(f"• Ancillas:   {args.ancillas}")
    print(f"• Shots:      {args.shots}")
    print(f"• t:          {args.t}")
    print(f"• Trotter:    {args.trotter_steps}")

    # Noise summary
    noise_params = None
    if args.noisy:
        noise_params = {
            "p_dep": args.p_dep,
            "p_amp": args.p_amp,
        }
        print(f"• Noise:      dep={args.p_dep}, amp={args.p_amp}")
    else:
        print("• Noise:      OFF")

    # Hamiltonian + HF state
    symbols = cfg["symbols"]
    coords = cfg["coordinates"]
    charge = cfg["charge"]
    basis = cfg["basis"]

    start_time = time.time()
    H, n_qubits = build_hamiltonian(symbols, coords, charge, basis)

    electrons = infer_electrons(symbols, charge)
    hf_state = qchem.hf_state(electrons, n_qubits)

    # Caching
    sig = signature_hash(
        molecule=args.molecule,
        n_ancilla=args.ancillas,
        t=args.t,
        noise=bool(noise_params),
        shots=args.shots,
    )

    cached = None if args.force else load_qpe_result(args.molecule, sig)

    if cached is not None:
        print("\n📂 Loaded cached result.")
        result = cached
    else:
        print("\n▶️ Running new QPE simulation...")
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

    elapsed = time.time() - start_time

    # Summary
    print("\n✅ QPE completed.")
    print(f"Most probable state : {result['best_bitstring']}")
    print(f"Estimated energy    : {result['energy']:.8f} Ha")
    print(f"Hartree–Fock energy : {result['hf_energy']:.8f} Ha")
    print(f"ΔE (QPE − HF)       : {result['energy'] - result['hf_energy']:+.8f} Ha")
    print(f"⏱  Elapsed          : {elapsed:.2f}s")
    print(f"Total qubits        : system={n_qubits}, ancillas={args.ancillas}")

    # Plot
    if args.plot or args.save_plot:
        plot_qpe_distribution(
            result,
            show=args.plot,
            save=args.save_plot,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  QPE simulation interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
