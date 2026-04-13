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
    python -m qpe --molecule H2 --ancillas 4 --t 1.0 --shots 1000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from qpe.core import run_qpe
from qpe.io_utils import ensure_dirs
from qpe.visualize import plot_qpe_distribution


def _parse_symbols(s: str | None) -> list[str] | None:
    if s is None or str(s).strip() == "":
        return None
    return [tok.strip() for tok in str(s).split(",") if tok.strip()]


def _parse_coordinates(s: str | None) -> np.ndarray | None:
    if s is None or str(s).strip() == "":
        return None

    rows: list[list[float]] = []
    for chunk in str(s).split(";"):
        part = chunk.strip()
        if not part:
            continue

        if "," in part:
            vals = [x.strip() for x in part.split(",") if x.strip()]
        else:
            vals = [x.strip() for x in part.split() if x.strip()]

        if len(vals) != 3:
            raise ValueError(
                "Each coordinate row must contain exactly 3 values. "
                "Example: --coordinates '0,0,0; 0,0,0.74'"
            )

        rows.append([float(v) for v in vals])

    if not rows:
        return None

    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Coordinates must parse to an array of shape (N, 3).")
    return arr


def _validated_geometry_inputs(
    args: argparse.Namespace,
) -> tuple[list[str] | None, np.ndarray | None]:
    symbols = _parse_symbols(getattr(args, "symbols", None))
    coordinates = _parse_coordinates(getattr(args, "coordinates", None))

    if (symbols is None) ^ (coordinates is None):
        raise ValueError(
            "Explicit geometry mode requires both --symbols and --coordinates."
        )

    if symbols is not None and coordinates is not None:
        if len(symbols) != int(len(coordinates)):
            raise ValueError(
                f"Mismatch between symbols ({len(symbols)}) and coordinate rows "
                f"({len(coordinates)})."
            )

    return symbols, coordinates


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
        default="H2",
        help="Molecule identifier (static registry key or parametric tag, e.g. H2, LiH, H2O_ANGLE, H2_BOND)",
    )

    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated atomic symbols for explicit geometry mode, e.g. 'H,H'",
    )
    parser.add_argument(
        "--coordinates",
        type=str,
        default=None,
        help="Semicolon-separated xyz rows for explicit geometry mode, e.g. '0,0,0; 0,0,0.74'",
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="sto-3g",
        help="Basis set for explicit geometry mode",
    )
    parser.add_argument(
        "--charge",
        type=int,
        default=0,
        help="Total molecular charge for explicit geometry mode",
    )
    parser.add_argument(
        "--active-electrons",
        type=int,
        default=None,
        help="Optional active-space electron count for chemistry Hamiltonian construction.",
    )
    parser.add_argument(
        "--active-orbitals",
        type=int,
        default=None,
        help="Optional active-space spatial orbital count for chemistry Hamiltonian construction.",
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
        help="Coordinate input unit for geometry values (allowed: angstrom, bohr). Energies are always reported in Hartree (Ha).",
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
        default=1000,
        help="Number of measurement shots",
    )

    # Noise model
    parser.add_argument("--noisy", action="store_true", help="Enable noise model")
    parser.add_argument(
        "--depolarizing-prob", type=float, default=0.0, help="Depolarizing probability"
    )
    parser.add_argument(
        "--amplitude-damping-prob",
        type=float,
        default=0.0,
        help="Amplitude damping probability",
    )
    parser.add_argument(
        "--phase-damping-prob",
        type=float,
        default=0.0,
        help="Phase damping probability",
    )
    parser.add_argument(
        "--bit-flip-prob", type=float, default=0.0, help="Bit-flip probability"
    )
    parser.add_argument(
        "--phase-flip-prob", type=float, default=0.0, help="Phase-flip probability"
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
    symbols, coordinates = _validated_geometry_inputs(args)
    molecule_label = (
        str(args.molecule).strip()
        if symbols is None
        else (str(args.molecule).strip() or "molecule")
    )

    print("\n🧮  QPE Simulation")
    print(f"• Molecule:   {molecule_label}")
    if symbols is not None and coordinates is not None:
        print(f"• Symbols:    {','.join(symbols)}")
        print(f"• Charge:     {int(args.charge)}")
        print(f"• Basis:      {args.basis}")
    print(f"• Mapping:    {args.mapping}")
    print(f"• Unit:       {args.unit}")
    print(f"• Ancillas:   {args.ancillas}")
    print(f"• Shots:      {args.shots}")
    print(f"• t:          {args.t}")
    print(f"• Trotter:    {args.trotter_steps}")
    print(f"• Seed:       {args.seed}")

    # ------------------------------------------------------------
    # Caching (hash depends on run-relevant QPE parameters)
    # ------------------------------------------------------------
    print("\n▶️ Running QPE (will use cache unless --force)...")
    start_time = time.time()

    # Let qpe.core handle caching + saving (single responsibility)
    result = run_qpe(
        molecule=molecule_label,
        seed=int(args.seed),
        n_ancilla=int(args.ancillas),
        t=float(args.t),
        trotter_steps=int(args.trotter_steps),
        shots=int(args.shots),
        plot=False,
        noisy=bool(args.noisy),
        depolarizing_prob=float(args.depolarizing_prob),
        amplitude_damping_prob=float(args.amplitude_damping_prob),
        phase_damping_prob=float(args.phase_damping_prob),
        bit_flip_prob=float(args.bit_flip_prob),
        phase_flip_prob=float(args.phase_flip_prob),
        force=bool(args.force),
        symbols=symbols,
        coordinates=coordinates,
        basis=str(args.basis),
        charge=int(args.charge),
        active_electrons=(
            None
            if getattr(args, "active_electrons", None) is None
            else int(args.active_electrons)
        ),
        active_orbitals=(
            None
            if getattr(args, "active_orbitals", None) is None
            else int(args.active_orbitals)
        ),
        mapping=str(args.mapping),
        unit=str(args.unit),
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

    sys_n = int(result.get("num_qubits", -1))
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
        raise SystemExit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        raise SystemExit(1)
