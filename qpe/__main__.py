"""
qpe.__main__
============
Command-line interface for Quantum Phase Estimation (QPE).

This CLI mirrors the VQE CLI philosophy:
    • clean argument parsing
    • cached result loading
    • no circuit logic mixed with plotting
    • single Hamiltonian pipeline via qpe.hamiltonian (which delegates to common)

Example:
    python -m qpe --molecule H2 --ancillas 4 --t 1.0 --shots 2000
"""

from __future__ import annotations

import argparse
import time

from qpe.core import run_qpe
from qpe.hamiltonian import build_hamiltonian
from qpe.io_utils import ensure_dirs
from qpe.visualize import plot_qpe_distribution


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        prog="qpe",
        description="Quantum Phase Estimation (QPE) simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducible caching and simulation",
    )

    parser.add_argument(
        "-m",
        "--molecule",
        required=True,
        help="Molecule identifier (static registry key or parametric tag, e.g. H2, LiH, H2O_ANGLE, H2_BOND)",
    )

    parser.add_argument(
        "--mapping",
        type=str,
        default="jordan_wigner",
        help="Fermion-to-qubit mapping (best-effort). Examples: jordan_wigner, bravyi_kitaev, parity",
    )

    parser.add_argument(
        "--unit",
        type=str,
        default="angstrom",
        help="Coordinate unit passed through to Hamiltonian construction (e.g., angstrom, bohr)",
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
    print(f"• Mapping:    {args.mapping}")
    print(f"• Unit:       {args.unit}")
    print(f"• Ancillas:   {args.ancillas}")
    print(f"• Shots:      {args.shots}")
    print(f"• t:          {args.t}")
    print(f"• Trotter:    {args.trotter_steps}")
    print(f"• Seed:       {args.seed}")

    noise_params = None
    if args.noisy:
        noise_params = {"p_dep": float(args.p_dep), "p_amp": float(args.p_amp)}
        print(f"• Noise:      dep={args.p_dep}, amp={args.p_amp}")
    else:
        print("• Noise:      OFF")

    # ------------------------------------------------------------
    # Caching (hash depends on run-relevant QPE parameters)
    # ------------------------------------------------------------
    print("\n▶️ Running QPE (will use cache unless --force)...")
    start_time = time.time()

    # Build Hamiltonian + HF state via the unified pipeline
    (
        H,
        n_qubits,
        hf_state,
        _symbols,
        _coordinates,
        _basis,
        _charge,
        _unit_out,
    ) = build_hamiltonian(
        str(args.molecule),
        mapping=str(args.mapping),
        unit=str(args.unit),
    )

    # Let qpe.core handle caching + saving (single responsibility)
    result = run_qpe(
        hamiltonian=H,
        hf_state=hf_state,
        seed=int(args.seed),
        n_ancilla=int(args.ancillas),
        t=float(args.t),
        trotter_steps=int(args.trotter_steps),
        noise_params=noise_params,
        shots=int(args.shots),
        molecule_name=str(args.molecule),
        system_qubits=int(n_qubits),
        mapping=str(args.mapping),
        unit=str(args.unit),
        force=bool(args.force),
    )

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
