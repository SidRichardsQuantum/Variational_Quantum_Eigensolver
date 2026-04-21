"""
qite.__main__
-------------

CLI entrypoint for VarQITE / VarQRTE routines.

Usage
-----
python -m qite --help

Commands
--------
run
    True VarQITE (McLachlan) parameter updates (pure-state only; noiseless).

run-qrte
    True VarQRTE (McLachlan real-time) parameter updates (pure-state only; noiseless).

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
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def _np():
    import numpy as np

    return np


def run_qite(*args, **kwargs):
    from qite.core import run_qite as _run_qite

    return _run_qite(*args, **kwargs)


def run_qrte(*args, **kwargs):
    from qite.core import run_qrte as _run_qrte

    return _run_qrte(*args, **kwargs)


def build_ansatz(*args, **kwargs):
    from qite.engine import build_ansatz as _build_ansatz

    return _build_ansatz(*args, **kwargs)


def make_device(*args, **kwargs):
    from qite.engine import make_device as _make_device

    return _make_device(*args, **kwargs)


def make_energy_qnode(*args, **kwargs):
    from qite.engine import make_energy_qnode as _make_energy_qnode

    return _make_energy_qnode(*args, **kwargs)


def make_state_qnode(*args, **kwargs):
    from qite.engine import make_state_qnode as _make_state_qnode

    return _make_state_qnode(*args, **kwargs)


def build_hamiltonian(*args, **kwargs):
    from qite.hamiltonian import build_hamiltonian as _build_hamiltonian

    return _build_hamiltonian(*args, **kwargs)


def _parse_int_list(s: Optional[str]) -> Optional[list[int]]:
    if s is None or str(s).strip() == "":
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_float_list(s: Optional[str]) -> Optional[list[float]]:
    if s is None or str(s).strip() == "":
        return None
    return [float(x.strip()) for x in str(s).split(",") if x.strip()]


def _parse_symbols(s: Optional[str]) -> Optional[list[str]]:
    if s is None or str(s).strip() == "":
        return None
    return [tok.strip() for tok in str(s).split(",") if tok.strip()]


def _parse_coordinates(s: Optional[str]) -> Optional[np.ndarray]:
    """
    Parse coordinates from a CLI string of the form:

        "0,0,0; 0,0,0.74"
        "0 0 0; 0 0 0.74"

    Returns
    -------
    np.ndarray of shape (N, 3)
    """
    np = _np()

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


def _explicit_geometry_requested(args) -> bool:
    return (
        getattr(args, "symbols", None) is not None
        or getattr(args, "coordinates", None) is not None
    )


def _validated_geometry_inputs(
    args,
) -> tuple[Optional[list[str]], Optional[np.ndarray]]:
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


def _active_space_kwargs(args) -> dict[str, int | None]:
    return {
        "active_electrons": (
            None
            if getattr(args, "active_electrons", None) is None
            else int(args.active_electrons)
        ),
        "active_orbitals": (
            None
            if getattr(args, "active_orbitals", None) is None
            else int(args.active_orbitals)
        ),
    }


def _builtin_noise_from_args(args) -> dict[str, float]:
    return {
        "depolarizing": float(args.depolarizing_prob),
        "amplitude_damping": float(args.amplitude_damping_prob),
        "phase_damping": float(args.phase_damping_prob),
        "bit_flip": float(args.bit_flip_prob),
        "phase_flip": float(args.phase_flip_prob),
    }


def _format_noise_summary(noise: dict[str, float]) -> str:
    labels = {
        "depolarizing": "dep",
        "amplitude_damping": "amp",
        "phase_damping": "phase",
        "bit_flip": "bit",
        "phase_flip": "phase_flip",
    }
    parts = []
    for key, label in labels.items():
        val = float(noise.get(key, 0.0))
        if val > 0.0:
            parts.append(f"{label}={val:g}")
    return ", ".join(parts)


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
    run_p.add_argument("--steps", type=int, default=75)
    run_p.add_argument("--dtau", type=float, default=0.2)
    run_p.add_argument("--seed", type=int, default=0)

    run_p.add_argument("--basis", type=str, default="sto-3g")
    run_p.add_argument("--charge", type=int, default=0)
    run_p.add_argument("--active-electrons", type=int, default=None)
    run_p.add_argument("--active-orbitals", type=int, default=None)
    run_p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated atomic symbols for explicit geometry mode, e.g. 'H,H'",
    )
    run_p.add_argument(
        "--coordinates",
        type=str,
        default=None,
        help="Semicolon-separated xyz rows for explicit geometry mode, e.g. '0,0,0; 0,0,0.74'",
    )

    run_p.add_argument("--mapping", type=str, default="jordan_wigner")
    run_p.add_argument(
        "--unit",
        type=str,
        default="angstrom",
        help="Coordinate input unit for geometry values (allowed: angstrom, bohr). Energies are always reported in Hartree (Ha).",
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
    # True VarQRTE run (noiseless)
    # -----------------------------------------------------------------
    qrte_p = sub.add_parser("run-qrte", help="Run true VarQRTE (noiseless; cached).")
    qrte_p.add_argument("--molecule", type=str, default="H2")
    qrte_p.add_argument("--ansatz", type=str, default="UCCSD")
    qrte_p.add_argument("--steps", type=int, default=50)
    qrte_p.add_argument("--dt", type=float, default=0.05)
    qrte_p.add_argument("--seed", type=int, default=0)
    qrte_p.add_argument("--basis", type=str, default="sto-3g")
    qrte_p.add_argument("--charge", type=int, default=0)
    qrte_p.add_argument("--active-electrons", type=int, default=None)
    qrte_p.add_argument("--active-orbitals", type=int, default=None)
    qrte_p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated atomic symbols for explicit geometry mode, e.g. 'H,H'",
    )
    qrte_p.add_argument(
        "--coordinates",
        type=str,
        default=None,
        help="Semicolon-separated xyz rows for explicit geometry mode, e.g. '0,0,0; 0,0,0.74'",
    )
    qrte_p.add_argument("--mapping", type=str, default="jordan_wigner")
    qrte_p.add_argument(
        "--unit",
        type=str,
        default="angstrom",
        help="Coordinate input unit for geometry values (allowed: angstrom, bohr). Energies are always reported in Hartree (Ha).",
    )
    qrte_p.add_argument("--plot", action="store_true", help="Generate plots.")
    qrte_p.add_argument("--no-plot", action="store_true", help="Disable plots.")
    qrte_p.add_argument("--show", action="store_true", help="Show plots.")
    qrte_p.add_argument("--no-show", action="store_true", help="Do not show plots.")
    qrte_p.add_argument("--force", action="store_true", help="Ignore cache and rerun.")
    qrte_p.add_argument("--fd-eps", type=float, default=1e-3)
    qrte_p.add_argument("--reg", type=float, default=1e-6)
    qrte_p.add_argument(
        "--solver", type=str, default="solve", choices=["solve", "lstsq", "pinv"]
    )
    qrte_p.add_argument("--pinv-rcond", type=float, default=1e-10)

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
        default=75,
        help="VarQITE steps used to converge parameters.",
    )
    ev_p.add_argument("--dtau", type=float, default=0.2)
    ev_p.add_argument("--seed", type=int, default=0)

    ev_p.add_argument("--basis", type=str, default="sto-3g")
    ev_p.add_argument("--charge", type=int, default=0)
    ev_p.add_argument("--active-electrons", type=int, default=None)
    ev_p.add_argument("--active-orbitals", type=int, default=None)
    ev_p.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated atomic symbols for explicit geometry mode, e.g. 'H,H'",
    )
    ev_p.add_argument(
        "--coordinates",
        type=str,
        default=None,
        help="Semicolon-separated xyz rows for explicit geometry mode, e.g. '0,0,0; 0,0,0.74'",
    )

    ev_p.add_argument("--mapping", type=str, default="jordan_wigner")
    ev_p.add_argument(
        "--unit",
        type=str,
        default="angstrom",
        help="Coordinate input unit for geometry values (allowed: angstrom, bohr). Energies are always reported in Hartree (Ha).",
    )

    ev_p.add_argument(
        "--depolarizing-prob", type=float, default=0.0, help="Depolarizing probability."
    )
    ev_p.add_argument(
        "--amplitude-damping-prob",
        type=float,
        default=0.0,
        help="Amplitude damping probability.",
    )
    ev_p.add_argument(
        "--phase-damping-prob",
        type=float,
        default=0.0,
        help="Phase damping probability.",
    )
    ev_p.add_argument(
        "--bit-flip-prob", type=float, default=0.0, help="Bit-flip probability."
    )
    ev_p.add_argument(
        "--phase-flip-prob", type=float, default=0.0, help="Phase-flip probability."
    )

    ev_p.add_argument(
        "--sweep-noise-type",
        type=str,
        default="depolarizing",
        choices=[
            "depolarizing",
            "amplitude_damping",
            "phase_damping",
            "bit_flip",
            "phase_flip",
        ],
        help="Built-in noise channel to sweep in multi-seed mode.",
    )
    ev_p.add_argument(
        "--sweep-levels",
        type=str,
        default=None,
        help="Comma-separated probability levels for the selected sweep noise type (e.g. 0,0.02,0.04).",
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

    out_g = ev_p.add_mutually_exclusive_group(required=False)
    out_g.add_argument(
        "--pretty", action="store_true", help="Print a human-readable summary."
    )
    out_g.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON output."
    )

    ev_p.add_argument("--fd-eps", type=float, default=1e-3)
    ev_p.add_argument("--reg", type=float, default=1e-6)
    ev_p.add_argument(
        "--solver", type=str, default="solve", choices=["solve", "lstsq", "pinv"]
    )
    ev_p.add_argument("--pinv-rcond", type=float, default=1e-10)

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
    symbols, coordinates = _validated_geometry_inputs(args)
    active_space = _active_space_kwargs(args)

    return run_qite(
        molecule=str(args.molecule),
        seed=int(args.seed),
        steps=int(args.steps),
        dtau=float(args.dtau),
        ansatz_name=str(args.ansatz),
        noisy=False,
        symbols=symbols,
        coordinates=coordinates,
        basis=str(args.basis),
        charge=int(args.charge),
        active_electrons=active_space["active_electrons"],
        active_orbitals=active_space["active_orbitals"],
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


def _run_varqrte(args) -> dict:
    plot, show = _resolve_plot_show(args)
    symbols, coordinates = _validated_geometry_inputs(args)
    active_space = _active_space_kwargs(args)

    return run_qrte(
        molecule=str(args.molecule),
        seed=int(args.seed),
        steps=int(args.steps),
        dt=float(args.dt),
        ansatz_name=str(args.ansatz),
        noisy=False,
        symbols=symbols,
        coordinates=coordinates,
        basis=str(args.basis),
        charge=int(args.charge),
        active_electrons=active_space["active_electrons"],
        active_orbitals=active_space["active_orbitals"],
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
    charge: int,
    active_electrons: int | None,
    active_orbitals: int | None,
    hf_state,
    ansatz: str,
    seed: int,
    theta: np.ndarray,
    dep: float,
    amp: float,
    phase: float,
    bit: float,
    phase_flip: float,
):
    """
    Evaluate Tr[rho H] under noise on default.mixed and also return diag(rho).
    """
    np = _np()

    dev = make_device(int(qubits), noisy=True)

    ansatz_fn, _ = build_ansatz(
        str(ansatz),
        int(qubits),
        seed=int(seed),
        symbols=symbols,
        coordinates=coordinates,
        charge=int(charge),
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
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
        phase_damping_prob=float(phase),
        bit_flip_prob=float(bit),
        phase_flip_prob=float(phase_flip),
        noise_model=None,
    )

    rho_q = make_state_qnode(
        dev,
        ansatz_fn,
        int(qubits),
        noisy=True,
        depolarizing_prob=float(dep),
        amplitude_damping_prob=float(amp),
        phase_damping_prob=float(phase),
        bit_flip_prob=float(bit),
        phase_flip_prob=float(phase_flip),
        noise_model=None,
    )

    E_val = float(E_q(theta))
    rho = np.array(rho_q(theta), dtype=complex)
    diag = np.clip(np.real(np.diag(rho)), 0.0, None)
    return E_val, diag


def _unpack_hamiltonian_metadata(
    *,
    molecule: str,
    mapping: str,
    unit: str,
    symbols: Optional[list[str]] = None,
    coordinates: Optional[np.ndarray] = None,
    basis: Optional[str] = None,
    charge: Optional[int] = None,
    active_electrons: Optional[int] = None,
    active_orbitals: Optional[int] = None,
):
    """
    Build Hamiltonian metadata in either registry mode or explicit geometry mode.

    Returns
    -------
    (H, n_qubits, hf_state, symbols, coordinates, basis, charge, mapping_out, unit_out)
    """
    np = _np()

    if symbols is not None and coordinates is not None:
        out = build_hamiltonian(
            molecule=None,
            symbols=list(symbols),
            coordinates=np.array(coordinates, dtype=float),
            charge=int(charge) if charge is not None else None,
            basis=str(basis) if basis is not None else None,
            active_electrons=(
                None if active_electrons is None else int(active_electrons)
            ),
            active_orbitals=(None if active_orbitals is None else int(active_orbitals)),
            mapping=str(mapping),
            unit=str(unit),
        )
    else:
        out = build_hamiltonian(
            molecule=str(molecule),
            active_electrons=(
                None if active_electrons is None else int(active_electrons)
            ),
            active_orbitals=(None if active_orbitals is None else int(active_orbitals)),
            mapping=str(mapping),
            unit=str(unit),
        )

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

    if len(out) == 8:
        (
            H,
            qubits,
            hf_state,
            symbols_out,
            coordinates_out,
            basis_out,
            charge_out,
            unit_out,
        ) = out
        mapping_out = str(mapping).strip().lower()
        return (
            H,
            qubits,
            hf_state,
            symbols_out,
            coordinates_out,
            basis_out,
            charge_out,
            mapping_out,
            unit_out,
        )

    if len(out) >= 9:
        (
            H,
            qubits,
            hf_state,
            symbols_out,
            coordinates_out,
            basis_out,
            charge_out,
            mapping_out,
            unit_out,
        ) = out[:9]
        return (
            H,
            qubits,
            hf_state,
            symbols_out,
            coordinates_out,
            basis_out,
            charge_out,
            mapping_out,
            unit_out,
        )

    raise TypeError(
        f"build_hamiltonian returned {len(out)} values; expected 3, 8, or >=9."
    )


def eval_noise(args) -> dict:
    np = _np()

    symbols_in, coordinates_in = _validated_geometry_inputs(args)

    H, qubits, hf_state, symbols, coordinates, basis, charge, mapping_out, unit_out = (
        _unpack_hamiltonian_metadata(
            molecule=str(args.molecule),
            mapping=str(args.mapping),
            unit=str(args.unit),
            symbols=symbols_in,
            coordinates=coordinates_in,
            basis=str(args.basis),
            charge=int(args.charge),
            active_electrons=getattr(args, "active_electrons", None),
            active_orbitals=getattr(args, "active_orbitals", None),
        )
    )

    sweep_levels = _parse_float_list(getattr(args, "sweep_levels", None))
    sweep_noise_type = (
        str(getattr(args, "sweep_noise_type", "depolarizing")).strip().lower()
    )
    seeds = _parse_int_list(getattr(args, "seeds", None))
    if seeds is None:
        seeds = list(range(5))
    base_noise = _builtin_noise_from_args(args)
    active_space = _active_space_kwargs(args)

    def _get_noiseless_record(seed: int) -> dict:
        return run_qite(
            molecule=str(args.molecule),
            seed=int(seed),
            steps=int(args.steps),
            dtau=float(args.dtau),
            ansatz_name=str(args.ansatz),
            noisy=False,
            symbols=symbols_in,
            coordinates=coordinates_in,
            basis=str(args.basis),
            charge=int(args.charge),
            active_electrons=active_space["active_electrons"],
            active_orbitals=active_space["active_orbitals"],
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
            charge=int(charge),
            active_electrons=getattr(args, "active_electrons", None),
            active_orbitals=getattr(args, "active_orbitals", None),
            hf_state=np.array(hf_state, dtype=int),
            ansatz=str(args.ansatz),
            seed=int(args.seed),
            theta=theta,
            dep=float(base_noise["depolarizing"]),
            amp=float(base_noise["amplitude_damping"]),
            phase=float(base_noise["phase_damping"]),
            bit=float(base_noise["bit_flip"]),
            phase_flip=float(base_noise["phase_flip"]),
        )

        return {
            "mode": "single",
            "molecule": str(args.molecule).strip() or "molecule",
            "ansatz": str(args.ansatz),
            "seed": int(args.seed),
            "mapping": str(mapping_out),
            "unit": str(unit_out),
            "charge": int(charge),
            "basis": str(basis),
            "varqite_energy_noiseless": float(res["energy"]),
            "noisy_energy": float(E_val),
            "dep": float(base_noise["depolarizing"]),
            "amp": float(base_noise["amplitude_damping"]),
            "phase": float(base_noise["phase_damping"]),
            "bit_flip": float(base_noise["bit_flip"]),
            "phase_flip": float(base_noise["phase_flip"]),
            "diag": diag.tolist(),
        }

    per_seed_noiseless: dict[int, dict] = {}
    for sd in seeds:
        per_seed_noiseless[int(sd)] = _get_noiseless_record(int(sd))

    means: list[float] = []
    stds: list[float] = []
    per_level: list[dict] = []

    for p in sweep_levels:
        sweep_noise = dict(base_noise)
        sweep_noise[sweep_noise_type] = float(p)
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
                charge=int(charge),
                active_electrons=getattr(args, "active_electrons", None),
                active_orbitals=getattr(args, "active_orbitals", None),
                hf_state=np.array(hf_state, dtype=int),
                ansatz=str(args.ansatz),
                seed=int(sd),
                theta=th,
                dep=float(sweep_noise["depolarizing"]),
                amp=float(sweep_noise["amplitude_damping"]),
                phase=float(sweep_noise["phase_damping"]),
                bit=float(sweep_noise["bit_flip"]),
                phase_flip=float(sweep_noise["phase_flip"]),
            )
            Es.append(float(E_val))

        Es_arr = np.asarray(Es, dtype=float)
        mean = float(Es_arr.mean())
        std = float(Es_arr.std(ddof=1)) if len(Es_arr) > 1 else 0.0

        means.append(mean)
        stds.append(std)
        per_level.append(
            {
                "probability": float(p),
                "mean": mean,
                "std": std,
                "values": [float(x) for x in Es],
            }
        )

    return {
        "mode": "sweep_noise",
        "molecule": str(args.molecule).strip() or "molecule",
        "ansatz": str(args.ansatz),
        "mapping": str(mapping_out),
        "unit": str(unit_out),
        "charge": int(charge),
        "basis": str(basis),
        "seeds": [int(s) for s in seeds],
        "base_noise": base_noise,
        "sweep_noise_type": sweep_noise_type,
        "sweep_levels": [float(x) for x in sweep_levels],
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
    print(f"• Basis:    {out.get('basis')}")
    print(f"• Charge:   {out.get('charge')}")

    if mode == "single":
        print(f"• Seed:     {out.get('seed')}")
        print(
            f"• Noiseless VarQITE energy: {out.get('varqite_energy_noiseless'):+.10f} Ha"
        )
        print(f"• Noisy energy Tr[ρH]:      {out.get('noisy_energy'):+.10f} Ha")
        print(
            "• Noise: "
            f"dep={out.get('dep')}, "
            f"amp={out.get('amp')}, "
            f"phase={out.get('phase')}, "
            f"bit={out.get('bit_flip')}, "
            f"phase_flip={out.get('phase_flip')}"
        )
        return

    print(f"• Seeds:    {out.get('seeds')}")
    print(f"• Sweep:    {out.get('sweep_noise_type')}")
    print(f"• Base:     {_format_noise_summary(out.get('base_noise', {})) or 'none'}")
    print("• Noise sweep (mean ± std):")
    for p, m, s in zip(out["sweep_levels"], out["means"], out["stds"]):
        print(f"  p={p:0.3f} -> {m:+.10f} ± {s:.10f} Ha")


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval-noise":
        out = eval_noise(args)

        as_json = bool(getattr(args, "json", False))
        as_pretty = bool(getattr(args, "pretty", False))

        if as_json and as_pretty:
            raise ValueError("Choose only one of --json or --pretty.")

        if as_json:
            print(json.dumps(out, indent=2))
        else:
            _print_pretty_eval(out)
        return

    if args.command == "run-qrte":
        out = _run_varqrte(args)
        print(json.dumps(out, indent=2))
        return

    out = _run_varqite(args)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
