"""
qite.__main__
-------------

CLI entrypoint for VarQITE routines.

Usage
-----
python -m qite --help

Commands
--------
run
    True VarQITE (McLachlan) parameter updates (pure-state only; noiseless).

eval-noise
    Post-evaluate a converged VarQITE circuit under noise using default.mixed.
    Supports a single noise setting or a depolarizing sweep over multiple seeds.

Notes
-----
- VarQITE updates require a pure statevector; therefore `run` does not allow noise.
- Noise is supported only for evaluation of the converged parameters.
- Hamiltonians / HF state are sourced from qite.hamiltonian (which delegates to common).
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

import numpy as np

from qite.core import run_qite
from qite.engine import build_ansatz, make_device, make_energy_qnode, make_state_qnode
from qite.hamiltonian import build_hamiltonian


def _parse_int_list(s: Optional[str]) -> Optional[list[int]]:
    if s is None or str(s).strip() == "":
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_float_list(s: Optional[str]) -> Optional[list[float]]:
    if s is None or str(s).strip() == "":
        return None
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qite",
        description="VarQITE (McLachlan) imaginary-time solver and noisy evaluation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    sub = p.add_subparsers(dest="command", required=False)

    # -----------------------------------------------------------------
    # True VarQITE run (noiseless)
    # -----------------------------------------------------------------
    run_p = sub.add_parser("run", help="Run true VarQITE (noiseless; cached).")
    run_p.add_argument("--molecule", type=str, default="H2")
    run_p.add_argument("--ansatz", type=str, default="UCCSD")
    run_p.add_argument("--steps", type=int, default=50)
    run_p.add_argument("--dtau", type=float, default=0.2)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--mapping", type=str, default="jordan_wigner")
    run_p.add_argument(
        "--unit",
        type=str,
        default="angstrom",
        help="Coordinate unit passed through to Hamiltonian construction (e.g., angstrom, bohr)",
    )

    run_p.add_argument("--plot", action="store_true", help="Generate plots.")
    run_p.add_argument("--no-plot", action="store_true", help="Disable plots.")
    run_p.add_argument("--show", action="store_true", help="Show plots.")
    run_p.add_argument("--no-show", action="store_true", help="Do not show plots.")
    run_p.add_argument("--force", action="store_true", help="Ignore cache and rerun.")

    # VarQITE numerics (must match cache keys)
    run_p.add_argument("--fd-eps", type=float, default=1e-3)
    run_p.add_argument("--reg", type=float, default=1e-6)
    run_p.add_argument(
        "--solver", type=str, default="solve", choices=["solve", "lstsq", "pinv"]
    )
    run_p.add_argument("--pinv-rcond", type=float, default=1e-10)

    # -----------------------------------------------------------------
    # Noisy evaluation of converged parameters
    # -----------------------------------------------------------------
    ev_p = sub.add_parser(
        "eval-noise",
        help="Evaluate a converged VarQITE circuit under noise (default.mixed).",
    )
    ev_p.add_argument("--molecule", type=str, default="H2")
    ev_p.add_argument("--ansatz", type=str, default="UCCSD")
    ev_p.add_argument(
        "--steps",
        type=int,
        default=50,
        help="VarQITE steps used to converge parameters.",
    )
    ev_p.add_argument("--dtau", type=float, default=0.2)
    ev_p.add_argument("--seed", type=int, default=0)
    ev_p.add_argument("--mapping", type=str, default="jordan_wigner")
    ev_p.add_argument(
        "--unit",
        type=str,
        default="angstrom",
        help="Coordinate unit passed through to Hamiltonian construction (e.g., angstrom, bohr)",
    )

    ev_p.add_argument(
        "--dep", type=float, default=0.0, help="Depolarizing probability."
    )
    ev_p.add_argument(
        "--amp", type=float, default=0.0, help="Amplitude damping probability."
    )

    ev_p.add_argument(
        "--sweep-dep",
        type=str,
        default=None,
        help="Comma-separated depolarizing levels for a sweep (e.g. 0,0.02,0.04).",
    )
    ev_p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for sweep averaging (e.g. 0,1,2,3,4). Default is 0..4.",
    )

    ev_p.add_argument(
        "--force",
        action="store_true",
        help="Force refresh of VarQITE caches for requested seeds.",
    )

    # Output format: default to pretty, allow explicit JSON
    out_g = ev_p.add_mutually_exclusive_group(required=False)
    out_g.add_argument(
        "--pretty", action="store_true", help="Print a human-readable summary."
    )
    out_g.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON output."
    )

    # VarQITE numerics (must match cache keys)
    ev_p.add_argument("--fd-eps", type=float, default=1e-3)
    ev_p.add_argument("--reg", type=float, default=1e-6)
    ev_p.add_argument(
        "--solver", type=str, default="solve", choices=["solve", "lstsq", "pinv"]
    )
    ev_p.add_argument("--pinv-rcond", type=float, default=1e-10)

    # Default subcommand
    p.set_defaults(command="run")
    return p


def _resolve_plot_show(args):
    plot = True
    if getattr(args, "no_plot", False):
        plot = False
    if getattr(args, "plot", False):
        plot = True

    show = True
    if getattr(args, "no_show", False):
        show = False
    if getattr(args, "show", False):
        show = True

    return bool(plot), bool(show)


def _run_varqite(args) -> dict:
    plot, show = _resolve_plot_show(args)

    # VarQITE run is noiseless by design.
    return run_qite(
        molecule=str(args.molecule),
        seed=int(args.seed),
        steps=int(args.steps),
        dtau=float(args.dtau),
        ansatz_name=str(args.ansatz),
        noisy=False,
        mapping=str(args.mapping),
        unit=str(args.unit),
        plot=bool(plot),
        show=bool(show),
        force=bool(args.force),
        fd_eps=float(args.fd_eps),
        reg=float(args.reg),
        solver=str(args.solver),
        pinv_rcond=float(args.pinv_rcond),
    )


def _noisy_eval_energy_and_diag(
    *,
    H,
    qubits: int,
    symbols,
    coordinates,
    basis: str,
    hf_state,
    ansatz: str,
    seed: int,
    theta: np.ndarray,
    dep: float,
    amp: float,
):
    """
    Evaluate Tr[rho H] under noise on default.mixed and also return diag(rho).
    """
    dev = make_device(int(qubits), noisy=True)

    ansatz_fn, _ = build_ansatz(
        str(ansatz),
        int(qubits),
        seed=int(seed),
        symbols=symbols,
        coordinates=coordinates,
        basis=str(basis).strip().lower(),
        requires_grad=False,
        hf_state=hf_state,
    )

    E_q = make_energy_qnode(
        H,
        dev,
        ansatz_fn,
        int(qubits),
        noisy=True,
        depolarizing_prob=float(dep),
        amplitude_damping_prob=float(amp),
        noise_model=None,
    )

    rho_q = make_state_qnode(
        dev,
        ansatz_fn,
        int(qubits),
        noisy=True,
        depolarizing_prob=float(dep),
        amplitude_damping_prob=float(amp),
        noise_model=None,
    )

    E_val = float(E_q(theta))
    rho = np.array(rho_q(theta), dtype=complex)
    diag = np.clip(np.real(np.diag(rho)), 0.0, None)
    return E_val, diag


def _unpack_hamiltonian_metadata(*, molecule: str, mapping: str, unit: str):
    """
    Unpack the outputs of qite.hamiltonian.build_hamiltonian in a robust way.

    Supported return shapes
    -----------------------
    A) (H, n_qubits, hf_state)
       -> NOT sufficient for eval-noise (needs metadata); raise.

    B) (H, n_qubits, hf_state, symbols, coordinates, basis, charge, unit_out)
       -> legacy metadata (no mapping_out); mapping_out will be taken from the input.

    C) (H, n_qubits, hf_state, symbols, coordinates, basis, charge, mapping_out, unit_out)
       -> preferred/modern form (current qite.hamiltonian in your tree).

    Returns
    -------
    (H, n_qubits, hf_state, symbols, coordinates, basis, charge, mapping_out, unit_out)
    """
    out = build_hamiltonian(str(molecule), mapping=str(mapping), unit=str(unit))

    if not isinstance(out, (tuple, list)) or len(out) < 3:
        raise TypeError(
            "build_hamiltonian(...) must return at least (H, n_qubits, hf_state)."
        )

    if len(out) == 3:
        raise TypeError(
            "qite eval-noise requires build_hamiltonian to return metadata "
            "(symbols, coordinates, basis, charge, mapping_out, unit_out). "
            "Update qite.hamiltonian to forward metadata from common."
        )

    # Legacy metadata shape: no mapping_out
    if len(out) == 8:
        H, qubits, hf_state, symbols, coordinates, basis, charge, unit_out = out
        mapping_out = str(mapping).strip().lower()
        return (
            H,
            qubits,
            hf_state,
            symbols,
            coordinates,
            basis,
            charge,
            mapping_out,
            unit_out,
        )

    # Preferred shape: mapping_out + unit_out
    if len(out) >= 9:
        (
            H,
            qubits,
            hf_state,
            symbols,
            coordinates,
            basis,
            charge,
            mapping_out,
            unit_out,
        ) = out[:9]
        return (
            H,
            qubits,
            hf_state,
            symbols,
            coordinates,
            basis,
            charge,
            mapping_out,
            unit_out,
        )

    # Any other length (4..7) is an error
    raise TypeError(
        f"build_hamiltonian returned {len(out)} values; expected 3, 8, or >=9."
    )


def _eval_noise(args) -> dict:
    # Single source of truth for Hamiltonian + HF + molecule metadata
    H, qubits, hf_state, symbols, coordinates, basis, charge, mapping_out, unit_out = (
        _unpack_hamiltonian_metadata(
            molecule=str(args.molecule),
            mapping=str(args.mapping),
            unit=str(args.unit),
        )
    )

    sweep_levels = _parse_float_list(getattr(args, "sweep_dep", None))
    seeds = _parse_int_list(getattr(args, "seeds", None))
    if seeds is None:
        seeds = list(range(5))

    # Helper: run or load VarQITE result for a given seed (cached by run_qite)
    def _get_noiseless_record(seed: int) -> dict:
        return run_qite(
            molecule=str(args.molecule),
            seed=int(seed),
            steps=int(args.steps),
            dtau=float(args.dtau),
            ansatz_name=str(args.ansatz),
            noisy=False,
            mapping=str(args.mapping),
            unit=str(args.unit),
            plot=False,
            show=False,
            force=bool(args.force),
            fd_eps=float(args.fd_eps),
            reg=float(args.reg),
            solver=str(args.solver),
            pinv_rcond=float(args.pinv_rcond),
        )

    # Single evaluation (no sweep)
    if sweep_levels is None:
        res = _get_noiseless_record(int(args.seed))

        theta_shape = tuple(res["final_params_shape"])
        theta = np.array(res["final_params"], dtype=float).reshape(theta_shape)

        E_val, diag = _noisy_eval_energy_and_diag(
            H=H,
            qubits=int(qubits),
            symbols=symbols,
            coordinates=coordinates,
            basis=str(basis),
            hf_state=np.array(hf_state, dtype=int),
            ansatz=str(args.ansatz),
            seed=int(args.seed),
            theta=theta,
            dep=float(args.dep),
            amp=float(args.amp),
        )

        return {
            "mode": "single",
            "molecule": str(args.molecule),
            "ansatz": str(args.ansatz),
            "seed": int(args.seed),
            "mapping": str(mapping_out),
            "unit": str(unit_out),
            "charge": int(charge),
            "varqite_energy_noiseless": float(res["energy"]),
            "noisy_energy": float(E_val),
            "dep": float(args.dep),
            "amp": float(args.amp),
            "diag": diag.tolist(),
        }

    # Sweep depolarizing levels (mean/std across seeds)
    per_seed_noiseless: dict[int, dict] = {}
    for sd in seeds:
        per_seed_noiseless[int(sd)] = _get_noiseless_record(int(sd))

    means: list[float] = []
    stds: list[float] = []
    per_level: list[dict] = []

    for p in sweep_levels:
        Es: list[float] = []
        for sd in seeds:
            r = per_seed_noiseless[int(sd)]
            th_shape = tuple(r["final_params_shape"])
            th = np.array(r["final_params"], dtype=float).reshape(th_shape)

            E_val, _diag = _noisy_eval_energy_and_diag(
                H=H,
                qubits=int(qubits),
                symbols=symbols,
                coordinates=coordinates,
                basis=str(basis),
                hf_state=np.array(hf_state, dtype=int),
                ansatz=str(args.ansatz),
                seed=int(sd),
                theta=th,
                dep=float(p),
                amp=float(args.amp),
            )
            Es.append(float(E_val))

        Es_arr = np.asarray(Es, dtype=float)
        mean = float(Es_arr.mean())
        std = float(Es_arr.std(ddof=1)) if len(Es_arr) > 1 else 0.0

        means.append(mean)
        stds.append(std)
        per_level.append(
            {
                "dep": float(p),
                "mean": mean,
                "std": std,
                "values": [float(x) for x in Es],
            }
        )

    return {
        "mode": "sweep_dep",
        "molecule": str(args.molecule),
        "ansatz": str(args.ansatz),
        "mapping": str(mapping_out),
        "unit": str(unit_out),
        "charge": int(charge),
        "seeds": [int(s) for s in seeds],
        "amp": float(args.amp),
        "dep_levels": [float(x) for x in sweep_levels],
        "means": means,
        "stds": stds,
        "per_level": per_level,
    }


def _print_pretty_eval(out: dict) -> None:
    mode = out.get("mode", "")
    print("\nVarQITE noisy evaluation")
    print(f"• Molecule: {out.get('molecule')}")
    print(f"• Ansatz:   {out.get('ansatz')}")
    print(f"• Mapping:  {out.get('mapping')}")
    print(f"• Unit:     {out.get('unit')}")

    if mode == "single":
        print(f"• Seed:     {out.get('seed')}")
        print(
            f"• Noiseless VarQITE energy: {out.get('varqite_energy_noiseless'):+.10f} Ha"
        )
        print(f"• Noisy energy Tr[ρH]:      {out.get('noisy_energy'):+.10f} Ha")
        print(f"• Noise: dep={out.get('dep')}, amp={out.get('amp')}")
        return

    print(f"• Seeds:    {out.get('seeds')}")
    print(f"• Amp:      {out.get('amp')}")
    print("• Dep sweep (mean ± std):")
    for p, m, s in zip(out["dep_levels"], out["means"], out["stds"]):
        print(f"  p={p:0.3f} -> {m:+.10f} ± {s:.10f} Ha")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval-noise":
        out = _eval_noise(args)

        as_json = bool(getattr(args, "json", False))
        as_pretty = bool(getattr(args, "pretty", False))

        if as_json and as_pretty:
            # Mutually exclusive group should prevent this, but keep it robust.
            raise ValueError("Choose only one of --json or --pretty.")

        if as_json:
            print(json.dumps(out, indent=2))
        else:
            _print_pretty_eval(out)
        return

    out = _run_varqite(args)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
