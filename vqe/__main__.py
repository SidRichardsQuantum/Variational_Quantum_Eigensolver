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
    result = run_vqe(args.molecule, args.steps, args.plot)
    print("\nFinal Result:")
    print(result)

if __name__ == "__main__":
    main()
