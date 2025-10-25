import argparse
import numpy as np
from .core import run_vqe
from vqe.visualize import plot_convergence
from vqe.core import run_vqe_noise_sweep, run_vqe_optimizer_comparison, run_vqe_ansatz_comparison, run_vqe_multi_seed_noise, run_vqe_geometry_scan


def main():
    parser = argparse.ArgumentParser(
        description="Run a Variational Quantum Eigensolver (VQE) simulation."
    )

    parser.add_argument(
        "--molecule", "-m",
        type=str,
        default="H2",
        help="Molecule to simulate (default: H2)"
    )

    parser.add_argument(
        "--ansatz", "-a",
        type=str,
        default="StronglyEntanglingLayers",
        help="Ansatz to use (default: StronglyEntanglingLayers)"
    )

    parser.add_argument(
        "--optimizer", "-o",
        type=str,
        default="GradientDescent",
        help="Optimizer to use (default: GradientDescent)"
    )

    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=50,
        help="Number of optimization steps (default: 50)"
    )

    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Plot energy convergence and save to /images"
    )

    parser.add_argument(
        "--noisy", 
        action="store_true",
        help="Enable noise simulation"
    )
    
    parser.add_argument(
        "--depolarizing-prob",
        type=float,
        default=0.0,
        help="Depolarizing noise probability per wire"
    )
    
    parser.add_argument(
        "--amplitude-damping-prob",
        type=float,
        default=0.0,
        help="Amplitude damping probability per wire"
    )
    
    parser.add_argument(
        "--compare-noise",
        action="store_true",
        help="Compare noiseless vs noisy runs"
    )

    parser.add_argument(
       "--noise-sweep",
        action="store_true",
        help="Run VQE for a range of noise levels and plot energy convergence for each.",
    )

    parser.add_argument(
        "--compare-optimizers",
        nargs="+",
        help="Compare multiple optimizers under the same noise conditions (e.g. Adam GradientDescent Momentum)."
    )

    parser.add_argument(
        "--compare-ansatzes",
        nargs="+",
        help="Compare multiple ansatzes under the same optimizer and noise conditions (e.g. RY-CZ Minimal TwoQubit-RY-CNOT)."
    )

    parser.add_argument(
        "--multi-seed-noise",
        action="store_true",
        help="Run noisy VQE over multiple random seeds and plot mean ± std energy/fidelity vs noise strength."
    )

    parser.add_argument(
        "--noise-type",
        type=str,
        choices=["depolarizing", "amplitude", "combined"],
        default="depolarizing",
        help="Type of noise to sweep in multi-seed runs (default: depolarizing)."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation even if cached results exist.",
    )

    parser.add_argument(
        "--scan-geometry",
        type=str,
        help="Perform a VQE geometry scan (e.g. H2_BOND or H2O_ANGLE)."
    )
    
    parser.add_argument(
        "--range",
        nargs=3,
        type=float,
        metavar=("START", "END", "NUM"),
        help="Parameter range for geometry scan, e.g. 100 110 5 for angles, or 0.5 1.5 10 for bond lengths."
    )
    
    parser.add_argument(
        "--param-name",
        type=str,
        default="angle",
        help="Geometry parameter to vary (angle or bond)."
    )

    args = parser.parse_args()

    if args.multi_seed_noise:
        run_vqe_multi_seed_noise(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=args.steps,
            noise_type=args.noise_type,
            force=args.force,
        )
        exit()

    if args.scan_geometry:
        start, end, num = args.range
        values = np.linspace(start, end, int(num))
        run_vqe_geometry_scan(
            molecule=args.scan_geometry,
            param_name=args.param_name,
            param_values=values,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=args.steps,
            force=args.force,
        )
        exit()

    if args.compare_optimizers:
        run_vqe_optimizer_comparison(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizers=args.compare_optimizers,
            steps=args.steps,
            noisy=args.noisy,
            depolarizing_prob=args.depolarizing_prob,
            amplitude_damping_prob=args.amplitude_damping_prob,
        )
        exit()

    if args.compare_ansatzes:
        run_vqe_ansatz_comparison(
            molecule=args.molecule,
            optimizer_name=args.optimizer,
            ansatzes=args.compare_ansatzes,
            steps=args.steps,
            noisy=args.noisy,
            depolarizing_prob=args.depolarizing_prob,
            amplitude_damping_prob=args.amplitude_damping_prob,
        )
        exit()

    if args.noise_sweep:
        run_vqe_noise_sweep(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=args.steps,
            force=args.force,
        )
        exit()

    if args.compare_noise:
        print(f"🔹 Comparing noisy vs noiseless VQE for {args.molecule}")
        # Run noiseless
        res_noiseless = run_vqe(
            args.molecule, args.steps, False,
            ansatz_name=args.ansatz, optimizer_name=args.optimizer,
            noisy=False
        )
        # Run noisy
        res_noisy = run_vqe(
            args.molecule, args.steps, False,
            ansatz_name=args.ansatz, optimizer_name=args.optimizer,
            noisy=True,
            depolarizing_prob=args.depolarizing_prob,
            amplitude_damping_prob=args.amplitude_damping_prob,
        )

        plot_convergence(
            res_noiseless["energies"],
            args.molecule,
            energies_noisy=res_noisy["energies"],
            optimizer=args.optimizer,
            ansatz=args.ansatz,
            dep_prob=args.depolarizing_prob,
            amp_prob=args.amplitude_damping_prob,
            noisy=True,
        )

        print("\n✅ Comparison complete.")
        return

    print(f"🔹 Running VQE for {args.molecule} with {args.steps} steps")
    print(f"   Ansatz: {args.ansatz} | Optimizer: {args.optimizer}")

    result = run_vqe(
        molecule=args.molecule,
        n_steps=args.steps,
        plot=args.plot,
        ansatz_name=args.ansatz,
        optimizer_name=args.optimizer,
        noisy=args.noisy,
        depolarizing_prob=args.depolarizing_prob,
        amplitude_damping_prob=args.amplitude_damping_prob,
        force=args.force,
    )


    # Clean up tensor display for CLI output
    def _to_serializable(obj):
        if hasattr(obj, "item"):
            try:
                return float(obj.item())
            except Exception:
                pass
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        return obj

    clean_result = _to_serializable(result)

    print("\nFinal Result:")
    print(clean_result)


if __name__ == "__main__":
    main()
