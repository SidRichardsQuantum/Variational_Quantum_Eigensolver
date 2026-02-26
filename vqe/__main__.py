"""
vqe.__main__.py

Command-line interface for the vqe package.

This file powers:

    $ python -m vqe ...

It supports:
    - Standard VQE
    - Noisy vs noiseless comparison
    - Noise sweeps
    - Optimizer comparison
    - Ansatz comparison
    - Multi-seed noise averaging
    - Geometry scans (bond length, bond angle)
    - Fermion-to-qubit mapping comparison
    - SSVQE for excited states
    - VQD for excited states

All CLI modes dispatch into vqe.core.* or vqe.ssvqe.run_ssvqe / vqe.vqd.run_vqd.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np

from vqe import (
    plot_convergence,
    run_adapt_vqe,
    run_eom_qse,
    run_eom_vqe,
    run_lr_vqe,
    run_qse,
    run_ssvqe,
    run_vqd,
    run_vqe,
    run_vqe_ansatz_comparison,
    run_vqe_geometry_scan,
    run_vqe_mapping_comparison,
    run_vqe_multi_seed_noise,
    run_vqe_optimizer_comparison,
)


def _parse_weights(
    raw: Optional[List[float]], *, num_states: int
) -> Optional[List[float]]:
    if raw is None:
        return None
    ws = [float(x) for x in raw]
    if len(ws) != int(num_states):
        raise ValueError(
            f"--weights must provide exactly num_states={num_states} values; got {len(ws)}"
        )
    if any(w <= 0 for w in ws):
        raise ValueError("--weights values must be strictly positive.")
    return ws


# ================================================================
# SPECIAL MODES DISPATCHER
# ================================================================
def handle_special_modes(args) -> bool:
    """
    Dispatch CLI options for all extended experiment modes.
    Returns True if a special mode handled the execution.
    """

    # ---------------------------
    #  SSVQE (excited states)
    # ---------------------------
    if args.ssvqe:
        print("🔹 Running SSVQE (excited states)...")
        if args.noisy and args.mapping != "jordan_wigner":
            print(
                "ℹ️  Note: SSVQE currently uses the package default mapping (jordan_wigner)."
            )

        weights = _parse_weights(args.weights, num_states=args.num_states)

        res = run_ssvqe(
            molecule=args.molecule,
            num_states=int(args.num_states),
            weights=weights,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=int(args.steps),
            stepsize=float(args.stepsize),
            seed=int(args.seed),
            noisy=bool(args.noisy),
            depolarizing_prob=float(args.depolarizing_prob),
            amplitude_damping_prob=float(args.amplitude_damping_prob),
            noise_model=None,
            reference_states=None,
            plot=bool(args.plot),
            force=bool(args.force),
        )

        finals = [float(traj[-1]) for traj in res["energies_per_state"]]
        print("Final energies per state (reported as E0, E1, ...):")
        for i, e in enumerate(finals):
            print(f"  E{i}: {e:+.10f} Ha")
        return True

    # ---------------------------
    #  LR-VQE (post-VQE excited states)
    # ---------------------------
    if args.lr_vqe:
        if args.noisy:
            raise ValueError(
                "LR-VQE is noiseless-only (statevector reference). Remove --noisy."
            )

        print("🔹 Running LR-VQE (post-VQE tangent-space TDA)...")

        res = run_lr_vqe(
            molecule=args.molecule,
            k=int(args.lr_k),
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=int(args.steps),
            stepsize=float(args.stepsize),
            seed=int(args.seed),
            mapping=str(args.mapping),
            fd_eps=float(args.lr_fd_eps),
            eps=float(args.lr_eps),
            force=bool(args.force),
            plot=bool(args.plot or args.save),
            show=bool(args.plot or args.save),
            save=bool(args.save),
        )

        print("LR-VQE excitations ω (lowest first):")
        for i, w in enumerate(res["excitations"]):
            print(f"  ω{i}: {float(w):+.10f} Ha")

        print("LR-VQE excited energies E0 + ω (lowest first):")
        for i, e in enumerate(res["eigenvalues"]):
            print(f"  E{i}: {float(e):+.10f} Ha")
        return True

    # ---------------------------
    #  EOM-VQE (post-VQE excited states; tangent full response)
    # ---------------------------
    if args.eom_vqe:
        if args.noisy:
            raise ValueError(
                "EOM-VQE is noiseless-only (statevector reference). Remove --noisy."
            )

        print("🔹 Running EOM-VQE (post-VQE tangent-space full response)...")

        res = run_eom_vqe(
            molecule=args.molecule,
            k=int(args.eom_k),
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=int(args.steps),
            stepsize=float(args.stepsize),
            seed=int(args.seed),
            mapping=str(args.mapping),
            fd_eps=float(args.eom_fd_eps),
            eps=float(args.eom_eps),
            omega_eps=float(args.eom_omega_eps),
            force=bool(args.force),
            plot=bool(args.plot or args.save),
            show=bool(args.plot or args.save),
            save=bool(args.save),
        )

        print("EOM-VQE excitations ω (lowest first):")
        for i, w in enumerate(res["excitations"]):
            print(f"  ω{i}: {float(w):+.10f} Ha")

        print("EOM-VQE excited energies E0 + ω (lowest first):")
        for i, e in enumerate(res["eigenvalues"]):
            print(f"  E{i}: {float(e):+.10f} Ha")
        return True

    # ---------------------------
    #  EOM-QSE (post-VQE commutator EOM in operator manifold)
    # ---------------------------
    if args.eom_qse:
        if args.noisy:
            raise ValueError(
                "EOM-QSE is noiseless-only (statevector reference). Remove --noisy."
            )

        print("🔹 Running EOM-QSE (post-VQE operator-manifold commutator EOM)...")

        res = run_eom_qse(
            molecule=args.molecule,
            k=int(args.eom_qse_k),
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=int(args.steps),
            stepsize=float(args.stepsize),
            seed=int(args.seed),
            mapping=str(args.mapping),
            pool=str(args.eom_qse_pool),
            max_ops=int(args.eom_qse_max_ops),
            eps=float(args.eom_qse_eps),
            imag_tol=float(args.eom_qse_imag_tol),
            omega_eps=float(args.eom_qse_omega_eps),
            force=bool(args.force),
        )

        print("EOM-QSE excitations ω (lowest first):")
        for i, w in enumerate(res["excitations"]):
            print(f"  ω{i}: {float(w):+.10f} Ha")

        print("EOM-QSE excited energies E0 + ω (lowest first):")
        for i, e in enumerate(res["eigenvalues"]):
            print(f"  E{i}: {float(e):+.10f} Ha")
        return True

    # ---------------------------
    #  VQD (excited states)
    # ---------------------------
    if args.vqd:
        print("🔹 Running VQD (excited states via deflation)...")
        if args.mapping != "jordan_wigner":
            print(
                "ℹ️  Note: VQD currently uses the package default mapping (jordan_wigner)."
            )

        res = run_vqd(
            molecule=args.molecule,
            num_states=int(args.num_states),
            beta=float(args.beta),
            beta_start=(
                args.beta_start if args.beta_start is None else float(args.beta_start)
            ),
            beta_ramp=str(args.beta_ramp),
            beta_hold_fraction=float(args.beta_hold_fraction),
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=int(args.steps),
            stepsize=float(args.stepsize),
            seed=int(args.seed),
            noisy=bool(args.noisy),
            depolarizing_prob=float(args.depolarizing_prob),
            amplitude_damping_prob=float(args.amplitude_damping_prob),
            noise_model=None,
            plot=bool(args.plot),
            force=bool(args.force),
        )

        finals = [float(traj[-1]) for traj in res["energies_per_state"]]
        print("Final energies per state (reported as E0, E1, ...):")
        for i, e in enumerate(finals):
            print(f"  E{i}: {e:+.10f} Ha")
        return True

    # ---------------------------
    #  ADAPT-VQE (adaptive ansatz)
    # ---------------------------
    if args.adapt:
        print("🔹 Running ADAPT-VQE (adaptive ansatz growth)...")

        inner_steps = (
            int(args.inner_steps) if args.inner_steps is not None else int(args.steps)
        )
        inner_stepsize = (
            float(args.inner_stepsize)
            if args.inner_stepsize is not None
            else float(args.stepsize)
        )

        res = run_adapt_vqe(
            molecule=args.molecule,
            pool=str(args.pool),
            max_ops=int(args.max_ops),
            grad_tol=float(args.grad_tol),
            inner_steps=inner_steps,
            inner_stepsize=inner_stepsize,
            optimizer_name=args.optimizer,
            seed=int(args.seed),
            noisy=bool(args.noisy),
            depolarizing_prob=float(args.depolarizing_prob),
            amplitude_damping_prob=float(args.amplitude_damping_prob),
            mapping=str(args.mapping),
            plot=bool(args.plot),
            force=bool(args.force),
        )
        return True

    # ---------------------------
    # Mapping comparison
    # ---------------------------
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

    # ---------------------------
    # Multi-seed noise sweep
    # ---------------------------
    if args.multi_seed_noise:
        run_vqe_multi_seed_noise(
            molecule=args.molecule,
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=args.steps,
            stepsize=args.stepsize,
            noise_type=args.noise_type,
            force=args.force,
            mapping=args.mapping,
        )
        return True

    # ---------------------------
    # Geometry scan
    # ---------------------------
    if args.scan_geometry:
        if args.range is None:
            raise ValueError("--scan-geometry requires --range START END NUM")
        start, end, num = args.range
        values = np.linspace(float(start), float(end), int(num))
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

    # ---------------------------
    # Optimizer comparison
    # ---------------------------
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
            mapping=args.mapping,
        )
        return True

    # ---------------------------
    # Ansatz comparison
    # ---------------------------
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
            mapping=args.mapping,
        )
        return True

    # ---------------------------
    #  QSE (post-VQE excited states)
    # ---------------------------
    if args.qse:
        print("🔹 Running QSE (post-VQE subspace expansion)...")

        res = run_qse(
            molecule=args.molecule,
            k=int(args.qse_k),
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            steps=int(args.steps),
            stepsize=float(args.stepsize),
            seed=int(args.seed),
            mapping=str(args.mapping),
            pool=str(args.qse_pool),
            max_ops=int(args.qse_max_ops),
            eps=float(args.qse_eps),
            force=bool(args.force),
        )

        print("QSE eigenvalues (lowest first):")
        for i, e in enumerate(res["eigenvalues"]):
            print(f"  E{i}: {float(e):+.10f} Ha")
        return True

    # ---------------------------
    # Compare noisy vs noiseless
    # ---------------------------
    if args.compare_noise:
        print(f"🔹 Comparing noisy vs noiseless VQE for {args.molecule}")

        res_noiseless = run_vqe(
            molecule=args.molecule,
            seed=int(args.seed),
            steps=int(args.steps),
            stepsize=float(args.stepsize),
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            noisy=False,
            mapping=args.mapping,
            force=bool(args.force),
            plot=False,
        )

        res_noisy = run_vqe(
            molecule=args.molecule,
            seed=int(args.seed),
            steps=int(args.steps),
            stepsize=float(args.stepsize),
            ansatz_name=args.ansatz,
            optimizer_name=args.optimizer,
            noisy=True,
            depolarizing_prob=float(args.depolarizing_prob),
            amplitude_damping_prob=float(args.amplitude_damping_prob),
            mapping=args.mapping,
            force=bool(args.force),
            plot=False,
        )

        plot_convergence(
            res_noiseless["energies"],
            args.molecule,
            energies_noisy=res_noisy["energies"],
            optimizer=args.optimizer,
            ansatz=args.ansatz,
            dep_prob=float(args.depolarizing_prob),
            amp_prob=float(args.amplitude_damping_prob),
            seed=int(args.seed),
        )
        return True

    return False


# ================================================================
# MAIN ENTRYPOINT
# ================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vqe",
        description="VQE / SSVQE / VQD Simulation Toolkit",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Core parameters
    # ------------------------------------------------------------------
    core = parser.add_argument_group("Core")
    core.add_argument(
        "-m",
        "--molecule",
        type=str,
        default="H2",
        help="Molecule (H2, LiH, H2O, H3+)",
    )
    core.add_argument("-a", "--ansatz", type=str, default="UCCSD", help="Ansatz name")
    core.add_argument(
        "-o", "--optimizer", type=str, default="Adam", help="Optimizer name"
    )
    core.add_argument(
        "-map",
        "--mapping",
        type=str,
        default="jordan_wigner",
        choices=["jordan_wigner", "bravyi_kitaev", "parity"],
        help="Fermion-to-qubit mapping (applies to VQE workflows; excited-state solvers currently default to jordan_wigner).",
    )
    core.add_argument(
        "-s",
        "--steps",
        type=int,
        default=50,
        help="Number of optimization iterations",
    )
    core.add_argument(
        "-lr",
        "--stepsize",
        type=float,
        default=0.2,
        help="Optimizer step size",
    )

    # ------------------------------------------------------------------
    # Noise controls
    # ------------------------------------------------------------------
    noise = parser.add_argument_group("Noise")
    noise.add_argument("--noisy", action="store_true", help="Enable noise")
    noise.add_argument("--depolarizing-prob", type=float, default=0.0)
    noise.add_argument("--amplitude-damping-prob", type=float, default=0.0)

    # ------------------------------------------------------------------
    # Experiment modes
    # ------------------------------------------------------------------
    exp = parser.add_argument_group("Modes")
    exp.add_argument("--compare-noise", action="store_true")
    exp.add_argument("--compare-optimizers", nargs="+")
    exp.add_argument("--compare-ansatzes", nargs="+")
    exp.add_argument("--multi-seed-noise", action="store_true")
    exp.add_argument(
        "--noise-type",
        type=str,
        choices=["depolarizing", "amplitude", "combined"],
        default="depolarizing",
    )
    exp.add_argument("--mapping-comparison", action="store_true")

    # ------------------------------------------------------------------
    # Geometry & excited-state solvers
    # ------------------------------------------------------------------
    special = parser.add_argument_group("Geometry / Excited States")
    special.add_argument(
        "--scan-geometry",
        type=str,
        help="Parametric geometry: H2_BOND, LiH_BOND, H2O_ANGLE",
    )
    special.add_argument(
        "--range",
        nargs=3,
        type=float,
        metavar=("START", "END", "NUM"),
        help="Geometry scan range (required with --scan-geometry)",
    )
    special.add_argument("--param-name", type=str, default="param")

    # Excited-state / adaptive solvers (mutually exclusive)
    excited = special.add_mutually_exclusive_group()
    excited.add_argument(
        "--ssvqe", action="store_true", help="Run SSVQE (excited states)"
    )
    excited.add_argument("--vqd", action="store_true", help="Run VQD (excited states)")
    excited.add_argument(
        "--adapt", action="store_true", help="Run ADAPT-VQE (adaptive ansatz)"
    )
    excited.add_argument(
        "--qse", action="store_true", help="Run QSE (post-VQE excited states)"
    )
    excited.add_argument(
        "--lr-vqe",
        action="store_true",
        help="Run LR-VQE (post-VQE tangent-space linear response; TDA, noiseless-only)",
    )
    excited.add_argument(
        "--eom-vqe",
        action="store_true",
        help="Run EOM-VQE (post-VQE tangent-space full response; noiseless-only)",
    )
    excited.add_argument(
        "--eom-qse",
        action="store_true",
        help="Run EOM-QSE (post-VQE commutator EOM in an operator manifold; noiseless-only)",
    )

    special.add_argument(
        "--num-states",
        type=int,
        default=2,
        help="Number of states for SSVQE/VQD (k-state)",
    )

    # SSVQE-specific knobs
    special.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="SSVQE weights w0 w1 ... (must provide exactly --num-states values). Default: [1,2,3,...].",
    )

    # VQD-specific knobs
    special.add_argument(
        "--beta", type=float, default=10.0, help="VQD deflation strength (beta_end)"
    )
    special.add_argument(
        "--beta-start",
        type=float,
        default=None,
        help="VQD beta_start (default: 0.0 if omitted)",
    )
    special.add_argument(
        "--beta-ramp",
        type=str,
        choices=["linear", "cosine"],
        default="linear",
        help="VQD beta ramp schedule",
    )
    special.add_argument(
        "--beta-hold-fraction",
        type=float,
        default=0.0,
        help="VQD hold fraction in [0,1) before ramping beta",
    )

    # QSE-specific knobs
    special.add_argument(
        "--qse-k",
        type=int,
        default=3,
        help="Number of QSE eigenvalues to return (lowest-k).",
    )
    special.add_argument(
        "--qse-pool",
        type=str,
        default="hamiltonian_topk",
        choices=["hamiltonian_topk"],
        help="QSE operator pool strategy.",
    )
    special.add_argument(
        "--qse-max-ops",
        type=int,
        default=24,
        help="Max operators in the QSE pool (includes identity).",
    )
    special.add_argument(
        "--qse-eps",
        type=float,
        default=1e-8,
        help="Overlap eigenvalue cutoff for QSE (S filtering).",
    )

    # ADAPT-VQE-specific knobs
    special.add_argument(
        "--pool",
        type=str,
        default="uccsd",
        choices=["uccsd", "uccs", "uccd"],
        help="ADAPT-VQE operator pool (default: uccsd).",
    )
    special.add_argument(
        "--max-ops",
        type=int,
        default=20,
        help="Maximum number of operators to add to the adaptive ansatz.",
    )
    special.add_argument(
        "--grad-tol",
        type=float,
        default=1e-3,
        help="Stop when the max operator gradient norm falls below this tolerance.",
    )
    special.add_argument(
        "--inner-steps",
        type=int,
        default=None,
        help="ADAPT-VQE inner-loop optimizer steps per outer iteration (default: --steps).",
    )
    special.add_argument(
        "--inner-stepsize",
        type=float,
        default=None,
        help="ADAPT-VQE inner-loop optimizer stepsize (default: --stepsize).",
    )

    # LR-VQE-specific knobs
    special.add_argument(
        "--lr-k",
        type=int,
        default=3,
        help="Number of LR-VQE excitation energies to return (lowest-k).",
    )
    special.add_argument(
        "--lr-fd-eps",
        type=float,
        default=1e-3,
        help="Finite-difference step for tangent vectors (δ).",
    )
    special.add_argument(
        "--lr-eps",
        type=float,
        default=1e-10,
        help="Overlap eigenvalue cutoff for LR-VQE (S filtering).",
    )
    special.add_argument(
        "--save",
        action="store_true",
        help="Save plots (in addition to showing them).",
    )

    # EOM-VQE-specific knobs
    special.add_argument(
        "--eom-k",
        type=int,
        default=3,
        help="Number of EOM-VQE excitation energies to return (lowest-k positive ω).",
    )
    special.add_argument(
        "--eom-fd-eps",
        type=float,
        default=1e-3,
        help="Finite-difference step for tangent vectors (δ).",
    )
    special.add_argument(
        "--eom-eps",
        type=float,
        default=1e-10,
        help="Overlap eigenvalue cutoff for EOM-VQE (S filtering).",
    )
    special.add_argument(
        "--eom-omega-eps",
        type=float,
        default=1e-12,
        help="Minimum ω to treat as a physical positive excitation.",
    )

    # EOM-QSE-specific knobs
    special.add_argument(
        "--eom-qse-k",
        type=int,
        default=3,
        help="Number of EOM-QSE excitation energies to return (lowest-k positive real-ish ω).",
    )
    special.add_argument(
        "--eom-qse-pool",
        type=str,
        default="hamiltonian_topk",
        choices=["hamiltonian_topk"],
        help="EOM-QSE operator pool strategy.",
    )
    special.add_argument(
        "--eom-qse-max-ops",
        type=int,
        default=24,
        help="Max operators in the EOM-QSE pool (includes identity).",
    )
    special.add_argument(
        "--eom-qse-eps",
        type=float,
        default=1e-8,
        help="Overlap eigenvalue cutoff for EOM-QSE (S filtering).",
    )
    special.add_argument(
        "--eom-qse-imag-tol",
        type=float,
        default=1e-10,
        help="Imaginary-part tolerance for accepting ω as real-ish.",
    )
    special.add_argument(
        "--eom-qse-omega-eps",
        type=float,
        default=1e-12,
        help="Require Re(ω) > omega_eps for acceptance.",
    )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    misc = parser.add_argument_group("Misc")
    misc.add_argument("--seed", type=int, default=0)
    misc.add_argument("--force", action="store_true", help="Ignore cached results")
    misc.add_argument("--plot", action="store_true", help="Plot convergence")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Summary banner
    # ------------------------------------------------------------------
    print("\n🧮  VQE Simulation")
    print(f"• Molecule:   {args.molecule}")
    print(f"• Ansatz:     {args.ansatz}")
    print(f"• Optimizer:  {args.optimizer}")
    print(f"• Mapping:    {args.mapping}")
    print(f"• Steps:      {args.steps}  | Stepsize: {args.stepsize}")
    print(f"• Noise:      {'ON' if args.noisy else 'OFF'}")
    print(f"• Seed:       {args.seed}")
    if args.ssvqe:
        print(f"• Mode:       SSVQE (num_states={args.num_states})")
    elif args.vqd:
        print(f"• Mode:       VQD   (num_states={args.num_states}, beta={args.beta})")
    elif args.qse:
        print(
            f"• Mode:       QSE (k={args.qse_k}, pool={args.qse_pool}, max_ops={args.qse_max_ops}, eps={args.qse_eps})"
        )
    elif args.adapt:
        inner_steps = args.inner_steps if args.inner_steps is not None else args.steps
        inner_stepsize = (
            args.inner_stepsize if args.inner_stepsize is not None else args.stepsize
        )
        print(
            f"• Mode:       ADAPT-VQE (pool={args.pool}, max_ops={args.max_ops}, grad_tol={args.grad_tol}, "
            f"inner_steps={inner_steps}, inner_stepsize={inner_stepsize})"
        )
    elif args.lr_vqe:
        print(
            f"• Mode:       LR-VQE (k={args.lr_k}, fd_eps={args.lr_fd_eps}, eps={args.lr_eps})"
        )
    elif args.eom_vqe:
        print(
            f"• Mode:       EOM-VQE (k={args.eom_k}, fd_eps={args.eom_fd_eps}, eps={args.eom_eps}, omega_eps={args.eom_omega_eps})"
        )
    elif args.eom_qse:
        print(
            f"• Mode:       EOM-QSE (k={args.eom_qse_k}, pool={args.eom_qse_pool}, max_ops={args.eom_qse_max_ops}, "
            f"eps={args.eom_qse_eps}, imag_tol={args.eom_qse_imag_tol}, omega_eps={args.eom_qse_omega_eps})"
        )

    # Try special modes first
    if handle_special_modes(args):
        return

    # ------------------------------------------------------------------
    # Default VQE run
    # ------------------------------------------------------------------
    print(f"🔹 Running standard VQE for {args.molecule}")
    result = run_vqe(
        molecule=args.molecule,
        steps=args.steps,
        stepsize=args.stepsize,
        ansatz_name=args.ansatz,
        optimizer_name=args.optimizer,
        mapping=args.mapping,
        noisy=args.noisy,
        depolarizing_prob=args.depolarizing_prob,
        amplitude_damping_prob=args.amplitude_damping_prob,
        force=args.force,
        plot=args.plot,
    )

    print("\nFinal result:")
    print({k: (float(v) if hasattr(v, "item") else v) for k, v in result.items()})


if __name__ == "__main__":
    main()
