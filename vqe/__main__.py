import argparse
from .core import run_vqe

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

    args = parser.parse_args()

    print(f"🔹 Running VQE for {args.molecule} with {args.steps} steps")
    print(f"   Ansatz: {args.ansatz} | Optimizer: {args.optimizer}")

    result = run_vqe(
        molecule=args.molecule,
        n_steps=args.steps,
        plot=args.plot,
        ansatz_name=args.ansatz,
        optimizer_name=args.optimizer
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
