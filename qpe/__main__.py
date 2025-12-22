"""
qpe.__main__
============
Command-line interface for Quantum Phase Estimation (QPE).

This CLI mirrors the VQE CLI philosophy:
    • clean argument parsing
    • cached result loading
    • no circuit logic mixed with plotting
    • single Hamiltonian pipeline via qpe.hamiltonian (which delegates to vqe_qpe_common)

Example:
    python -m qpe --molecule H2 --ancillas 4 --t 1.0 --shots 2000
"""

from __future__ import annotations

import argparse
import time

from qpe.core import run_qpe
from qpe.hamiltonian import build_hamiltonian
from qpe.io_utils import (
    ensure_dirs,
    load_qpe_result,
    save_qpe_result,
    signature_hash,
)
from qpe.visualize import plot_qpe_distribution
from vqe_qpe_common.molecules import MOLECULES


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Phase Estimation (QPE) simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--molecule",
        required=True,
        choices=sorted(MOLECULES.keys()),
        help="Molecule to simulate (from vqe_qpe_common.molecules.MOLECULES)",
    )

    parser.add_argument(
        "--ancillas",
        type=int,
        default=4,
        help="Number of ancilla qubits",
    )

    parser.add_argument(
        "--t",
        type=float,
        default=1.0,
        help="Evolution time in exp(-iHt)",
    )

    parser.add_argument(
        "--trotter-steps",
        type=int,
        default=2,
        help="Trotter steps for time evolution",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=2000,
        help="Number of measurement shots",
    )

    # Noise model
    parser.add_argument("--noisy", action="store_true", help="Enable noise model")
    parser.add_argument(
        "--p-dep", type=float, default=0.0, help="Depolarizing probability"
    )
    parser.add_argument(
        "--p-amp", type=float, default=0.0, help="Amplitude damping probability"
    )

    # Plotting
    parser.add_argument(
        "--plot", action="store_true", help="Show plot after simulation"
    )
    parser.add_argument(
        "--save-plot", action="store_true", help="Save QPE probability distribution"
    )

    parser.add_argument(
        "--force", action="store_true", help="Force rerun even if cached result exists"
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dirs()

    print("\n🧮  QPE Simulation")
    print(f"• Molecule:   {args.molecule}")
    print(f"• Ancillas:   {args.ancillas}")
    print(f"• Shots:      {args.shots}")
    print(f"• t:          {args.t}")
    print(f"• Trotter:    {args.trotter_steps}")

    noise_params = None
    if args.noisy:
        noise_params = {"p_dep": args.p_dep, "p_amp": args.p_amp}
        print(f"• Noise:      dep={args.p_dep}, amp={args.p_amp}")
    else:
        print("• Noise:      OFF")

    # ------------------------------------------------------------
    # Caching (hash only depends on run-relevant QPE parameters)
    # ------------------------------------------------------------
    sig = signature_hash(
        molecule=args.molecule,
        n_ancilla=args.ancillas,
        t=args.t,
        trotter_steps=args.trotter_steps,
        noise=noise_params,
        shots=args.shots,
    )

    cached = None if args.force else load_qpe_result(args.molecule, sig)
    if cached is not None:
        print("\n📂 Loaded cached result.")
        result = cached

        n_qubits = int(cached.get("system_qubits", -1))
        elapsed = 0.0
    else:
        print("\n▶️ Running new QPE simulation...")
        start_time = time.time()

        # Build Hamiltonian + HF state via the unified pipeline
        H, n_qubits, hf_state, symbols, coordinates, basis, charge = build_hamiltonian(
            args.molecule
        )

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

        result["system_qubits"] = int(n_qubits)

        save_qpe_result(result)
        elapsed = time.time() - start_time

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n✅ QPE completed.")
    print(f"Most probable state : {result['best_bitstring']}")
    print(f"Estimated energy    : {result['energy']:.8f} Ha")
    print(f"Hartree–Fock energy : {result['hf_energy']:.8f} Ha")
    print(f"ΔE (QPE − HF)       : {result['energy'] - result['hf_energy']:+.8f} Ha")
    if elapsed:
        print(f"⏱  Elapsed          : {elapsed:.2f}s")

    # Prefer the stored system_qubits field if present
    sys_n = int(result.get("system_qubits", -1))
    if sys_n >= 0:
        print(f"Total qubits        : system={sys_n}, ancillas={args.ancillas}")
    else:
        print(
            f"Total qubits        : ancillas={args.ancillas} (system qubits unknown in this record)"
        )

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    if args.plot or args.save_plot:
        plot_qpe_distribution(result, show=args.plot, save=args.save_plot)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹  QPE simulation interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
