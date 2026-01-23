"""
qite.__main__
-------------

CLI entrypoint for QITE routines.

Usage
-----
python -m qite --help

This mirrors the lightweight style of vqe.__main__ / qpe.__main__:
- Run a single cached QITE solve
- Optionally run a multi-seed noise study
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from qite.core import run_qite, run_qite_multi_seed_noise


def _parse_int_list(s: Optional[str]):
    if s is None or str(s).strip() == "":
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_float_list(s: Optional[str]):
    if s is None or str(s).strip() == "":
        return None
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qite",
        description="Imaginary-time / QITE-style solvers (PennyLane-based).",
    )

    sub = p.add_subparsers(dest="command", required=False)

    # -----------------------------------------------------------------
    # Default: single run (also available as `run`)
    # -----------------------------------------------------------------
    run_p = sub.add_parser("run", help="Run a single QITE solve (cached).")
    run_p.add_argument("--molecule", type=str, default="H2")
    run_p.add_argument("--ansatz", type=str, default="UCCSD")
    run_p.add_argument("--steps", type=int, default=50)
    run_p.add_argument("--dtau", type=float, default=0.2)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--mapping", type=str, default="jordan_wigner")

    run_p.add_argument("--noisy", action="store_true", help="Enable noise channels.")
    run_p.add_argument(
        "--dep", type=float, default=0.0, help="Depolarizing probability."
    )
    run_p.add_argument(
        "--amp", type=float, default=0.0, help="Amplitude damping probability."
    )

    run_p.add_argument("--plot", action="store_true", help="Generate plots.")
    run_p.add_argument("--no-plot", action="store_true", help="Disable plots.")
    run_p.add_argument("--show", action="store_true", help="Show plots.")
    run_p.add_argument("--no-show", action="store_true", help="Do not show plots.")
    run_p.add_argument("--force", action="store_true", help="Ignore cache and rerun.")

    # -----------------------------------------------------------------
    # Noise study: multi-seed stats
    # -----------------------------------------------------------------
    ns_p = sub.add_parser("noise", help="Run a multi-seed noise study.")
    ns_p.add_argument("--molecule", type=str, default="H2")
    ns_p.add_argument("--ansatz", type=str, default="UCCSD")
    ns_p.add_argument("--steps", type=int, default=30)
    ns_p.add_argument("--dtau", type=float, default=0.2)
    ns_p.add_argument("--mapping", type=str, default="jordan_wigner")

    ns_p.add_argument(
        "--noise-type",
        type=str,
        default="depolarizing",
        choices=["depolarizing", "amplitude", "combined"],
        help="Noise mode for the sweep.",
    )
    ns_p.add_argument(
        "--levels",
        type=str,
        default=None,
        help="Comma-separated noise levels (e.g. 0,0.02,0.04). Default is a small grid.",
    )
    ns_p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (e.g. 0,1,2,3,4). Default is 0..4.",
    )

    ns_p.add_argument("--show", action="store_true", help="Show plots.")
    ns_p.add_argument("--no-show", action="store_true", help="Do not show plots.")
    ns_p.add_argument("--force", action="store_true", help="Ignore cache and rerun.")

    # If user runs `python -m qite` without subcommand, treat it as `run`.
    p.set_defaults(command="run")

    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Plot/show flags
    plot = True
    if getattr(args, "plot", False):
        plot = True
    if getattr(args, "no_plot", False):
        plot = False

    show = True
    if getattr(args, "show", False):
        show = True
    if getattr(args, "no_show", False):
        show = False

    if args.command == "noise":
        seeds = _parse_int_list(getattr(args, "seeds", None))
        levels = _parse_float_list(getattr(args, "levels", None))

        out = run_qite_multi_seed_noise(
            molecule=str(args.molecule),
            ansatz_name=str(args.ansatz),
            steps=int(args.steps),
            dtau=float(args.dtau),
            seeds=seeds,
            noise_type=str(args.noise_type),
            noise_levels=levels,
            mapping=str(args.mapping),
            force=bool(args.force),
            show=bool(show),
        )
        print(json.dumps(out, indent=2))
        return

    # Default: single run
    out = run_qite(
        molecule=str(args.molecule),
        seed=int(args.seed),
        steps=int(args.steps),
        dtau=float(args.dtau),
        ansatz_name=str(args.ansatz),
        noisy=bool(getattr(args, "noisy", False)),
        depolarizing_prob=float(getattr(args, "dep", 0.0)),
        amplitude_damping_prob=float(getattr(args, "amp", 0.0)),
        mapping=str(args.mapping),
        plot=bool(plot),
        force=bool(getattr(args, "force", False)),
        show=bool(show),
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
