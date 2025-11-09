import argparse
import numpy as np
from .core import run_vqe
from vqe.visualize import plot_convergence
from vqe.core import (
    run_vqe_noise_sweep,
    run_vqe_optimizer_comparison,
    run_vqe_ansatz_comparison,
    run_vqe_multi_seed_noise,
    run_vqe_geometry_scan,
    run_vqe_mapping_comparison,
)


def handle_special_modes(args):
    """Handle all nonstandard experiment modes. Returns True if handled."""
    if args.ssvqe:
        from vqe.ssvqe import run_ssvqe
        res = run_ssvqe(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=args.steps,
            stepsize=args.stepsize,
            penalty_weight=args.penalty_weight,
            seed=args.seed,
            noisy=args.noisy,
            depolarizing_prob=args.depolarizing_prob,
            amplitude_damping_prob=args.amplitude_damping_prob,
            plot=args.plot,
            force=args.force,
        )
        print("\nFinal (SSVQE):")
        preview = {
            **res,
            "final_params": f"[{len(res['final_params'])} states; each {len(res['final_params'][0])} params]"
        }
        print(preview)
        return True

    if args.mapping_comparison:
        run_vqe_mapping_comparison(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=args.steps,
            stepsize=args.stepsize,
            seed=args.seed,
            force=args.force,
        )
        return True

    if args.multi_seed_noise:
        run_vqe_multi_seed_noise(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=args.steps,
            stepsize=args.stepsize,
            noise_type=args.noise_type,
            force=args.force,
        )
        return True

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
            stepsize=args.stepsize,
            force=args.force,
        )
        return True

    if args.compare_optimizers:
        run_vqe_optimizer_comparison(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizers=args.compare_optimizers,
            steps=args.steps,
            stepsize=args.stepsize,
            noisy=args.noisy,
            depolarizing_prob=args.depolarizing_prob,
            amplitude_damping_prob=args.amplitude_damping_prob,
        )
        return True

    if args.compare_ansatzes:
        run_vqe_ansatz_comparison(
            molecule=args.molecule,
            optimizer_name=args.optimizer,
            ansatzes=args.compare_ansatzes,
            steps=args.steps,
            stepsize=args.stepsize,
            noisy=args.noisy,
            depolarizing_prob=args.depolarizing_prob,
            amplitude_damping_prob=args.amplitude_damping_prob,
        )
        return True

    if args.noise_sweep:
        run_vqe_noise_sweep(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=args.steps,
            stepsize=args.stepsize,
            force=args.force,
        )
        return True

    if args.compare_noise:
        print(f"🔹 Comparing noisy vs noiseless VQE for {args.molecule}")
        res_noiseless = run_vqe(
            args.molecule, args.steps, False,
            stepsize=args.stepsize,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            noisy=False,
        )
        res_noisy = run_vqe(
            args.molecule, args.steps, False,
            stepsize=args.stepsize,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
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
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        prog="vqe",
        description="Variational Quantum Eigensolver (VQE) Simulation Toolkit",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # === Core simulation parameters ===
    core = parser.add_argument_group("Core Parameters")
    core.add_argument("-m", "--molecule",        type=str,   default="H2",          help="Molecule to simulate [default: H2]")
    core.add_argument("-a", "--ansatz",          type=str,   default="UCCSD",       help="Ansatz circuit [default: UCCSD]")
    core.add_argument("-o", "--optimizer",       type=str,   default="Adam",        help="Optimizer to use [default: Adam]")
    core.add_argument("-map", "--mapping",       type=str,   default="jordan_wigner",
                      choices=["jordan_wigner", "bravyi_kitaev", "parity"],
                      help="Fermion-to-qubit mapping [default: jordan_wigner]")
    core.add_argument("-s", "--steps",           type=int,   default=50,            help="Number of optimization steps [default: 50]")
    core.add_argument("-lr", "--stepsize",       type=float, default=0.2,           help="Learning rate / step size [default: 0.2]")

    # === Noise and environment ===
    noise = parser.add_argument_group("Noise & Environment")
    noise.add_argument("--noisy", action="store_true", help="Enable noise simulation")
    noise.add_argument("--depolarizing-prob", type=float, default=0.0, help="Depolarizing noise probability per wire [default: 0.0]")
    noise.add_argument("--amplitude-damping-prob", type=float, default=0.0, help="Amplitude damping probability per wire [default: 0.0]")

    # === Experiment modes ===
    exp = parser.add_argument_group("Experiment Modes")
    exp.add_argument("--compare-noise", action="store_true", help="Compare noiseless vs noisy runs")
    exp.add_argument("--noise-sweep", action="store_true", help="Run VQE for a range of noise levels")
    exp.add_argument("--compare-optimizers", nargs="+", help="Compare multiple optimizers (e.g. Adam GradientDescent)")
    exp.add_argument("--compare-ansatzes", nargs="+", help="Compare multiple ansatzes (e.g. RY-CZ Minimal)")
    exp.add_argument("--multi-seed-noise", action="store_true", help="Run multi-seed noise averaging")
    exp.add_argument("--noise-type", type=str, choices=["depolarizing", "amplitude", "combined"],
                     default="depolarizing", help="Noise type for multi-seed runs [default: depolarizing]")
    exp.add_argument("--mapping-comparison", action="store_true", help="Compare different fermion-to-qubit mappings")

    # === Geometry scan & SSVQE ===
    geom = parser.add_argument_group("Geometry & SSVQE")
    geom.add_argument("--scan-geometry", type=str, help="Perform a VQE geometry scan (e.g. H2_BOND)")
    geom.add_argument("--range", nargs=3, type=float, metavar=("START", "END", "NUM"), help="Range for geometry scan (e.g. 0.5 1.5 10)")
    geom.add_argument("--param-name", type=str, default="angle", help="Geometry parameter to vary [default: angle]")
    geom.add_argument("--ssvqe", action="store_true", help="Run two-state SSVQE (ground + first excited)")
    geom.add_argument("--penalty-weight", type=float, default=10.0, help="Penalty weight for |⟨ψ0|ψ1⟩|² term [default: 10.0]")

    # === Miscellaneous ===
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--seed", type=int, default=0, help="Random seed [default: 0]")
    misc.add_argument("-p", "--plot", action="store_true", help="Plot energy convergence and save to /images")
    misc.add_argument("--force", action="store_true", help="Force recomputation (ignore cached results)")

    args = parser.parse_args()

    # Summary header
    print("\n🧮  VQE Simulation Configuration")
    print(f"• Molecule:   {args.molecule}")
    print(f"• Ansatz:     {args.ansatz}")
    print(f"• Optimizer:  {args.optimizer}")
    print(f"• Mapping:    {args.mapping}")
    print(f"• Steps:      {args.steps}  | Stepsize: {args.stepsize}")
    print(f"• Noise:      {'ON' if args.noisy else 'OFF'}")
    print(f"• Seed:       {args.seed}")
    print()

    # Handle special experiment modes
    if handle_special_modes(args):
        return

    # === Default VQE run ===
    print(f"🔹 Running standard VQE for {args.molecule}")
    result = run_vqe(
        molecule=args.molecule,
        n_steps=args.steps,
        stepsize=args.stepsize,
        plot=args.plot,
        ansatz_name=args.ansatz,
        optimizer_name=args.optimizer,
        mapping=args.mapping,
        noisy=args.noisy,
        depolarizing_prob=args.depolarizing_prob,
        amplitude_damping_prob=args.amplitude_damping_prob,
        force=args.force,
    )

    # Clean print for CLI output
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
