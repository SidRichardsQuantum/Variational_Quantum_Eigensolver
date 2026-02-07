"""
vqe.adapt
---------

Adaptive Derivative-Assembled Pseudo-Trotter VQE (ADAPT-VQE).

This implementation is chemistry-oriented:
- reference state: Hartree–Fock determinant
- operator pool: UCC singles/doubles excitations (UCCS / UCCD / UCCSD)

Algorithm
----------------------
Iterate:
  1) (inner loop) optimize parameters for the current operator list
  2) (outer loop) score each remaining pool operator by |dE/dθ| at θ=0 if appended
  3) append the max-gradient operator if above tolerance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import pennylane as qml
from pennylane import numpy as np

from .ansatz import _build_ucc_data
from .engine import apply_optional_noise, build_optimizer, make_device
from .hamiltonian import build_hamiltonian
from .io_utils import (
    ensure_dirs,
    is_effectively_noisy,
    load_run_record,
    make_filename_prefix,
    make_run_config_dict,
    run_signature,
    save_run_record,
)


@dataclass(frozen=True)
class PoolOp:
    kind: str  # "single" | "double"
    wires: Tuple[int, ...]


def _make_ucc_pool(
    *,
    symbols: Sequence[str],
    coordinates,
    basis: str,
    pool: str,
) -> Tuple[List[PoolOp], np.ndarray]:
    singles, doubles, hf_state = _build_ucc_data(symbols, coordinates, basis=basis)

    pool_key = str(pool).strip().lower()
    if pool_key in {"uccsd", "ucc-sd", "sd", "both"}:
        ops = [PoolOp("single", tuple(ex)) for ex in singles] + [
            PoolOp("double", tuple(ex)) for ex in doubles
        ]
    elif pool_key in {"uccs", "ucc-s", "s", "singles"}:
        ops = [PoolOp("single", tuple(ex)) for ex in singles]
    elif pool_key in {"uccd", "ucc-d", "uccdoubles", "d", "doubles"}:
        ops = [PoolOp("double", tuple(ex)) for ex in doubles]
    else:
        raise ValueError("pool must be one of: 'uccsd', 'uccs', 'uccd'")

    return ops, np.array(hf_state, dtype=int)


def _apply_selected_ops(
    theta,
    *,
    hf_state: np.ndarray,
    ops: Sequence[PoolOp],
    num_wires: int,
) -> None:
    qml.BasisState(np.array(hf_state, dtype=int), wires=range(int(num_wires)))

    if len(ops) == 0:
        return

    if len(theta) != len(ops):
        raise ValueError(
            f"theta length {len(theta)} does not match ops length {len(ops)}"
        )

    for t, op in zip(theta, ops):
        if op.kind == "single":
            qml.SingleExcitation(t, wires=list(op.wires))
        elif op.kind == "double":
            qml.DoubleExcitation(t, wires=list(op.wires))
        else:
            raise ValueError(f"Unknown op kind: {op.kind!r}")


def _energy_qnode_factory(
    *,
    H,
    dev,
    hf_state: np.ndarray,
    ops: Sequence[PoolOp],
    num_wires: int,
    noisy: bool,
    depolarizing_prob: float,
    amplitude_damping_prob: float,
    noise_model: Optional[Callable[[list[int]], None]],
    diff_method: str,
):
    @qml.qnode(dev, diff_method=diff_method)
    def energy(theta):
        _apply_selected_ops(theta, hf_state=hf_state, ops=ops, num_wires=num_wires)
        apply_optional_noise(
            bool(noisy),
            float(depolarizing_prob),
            float(amplitude_damping_prob),
            int(num_wires),
            noise_model=noise_model,
        )
        return qml.expval(H)

    return energy


def _inner_optimize(
    *,
    energy_qnode,
    theta_init,
    optimizer_name: str,
    stepsize: float,
    steps: int,
) -> Tuple[np.ndarray, List[float]]:
    opt = build_optimizer(str(optimizer_name), stepsize=float(stepsize))

    theta = np.array(theta_init, requires_grad=True)
    energies: List[float] = [float(energy_qnode(theta))]

    for _ in range(int(steps)):
        try:
            theta, cost = opt.step_and_cost(energy_qnode, theta)
            e = float(cost)
        except AttributeError:
            theta = opt.step(energy_qnode, theta)
            e = float(energy_qnode(theta))
        energies.append(e)

    return theta, energies


def run_adapt_vqe(
    molecule: str = "H2",
    *,
    pool: str = "uccsd",
    max_ops: int = 20,
    grad_tol: float = 1e-3,
    inner_steps: int = 50,
    inner_stepsize: float = 0.2,
    optimizer_name: str = "Adam",
    seed: int = 0,
    mapping: str = "jordan_wigner",
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[[list[int]], None]] = None,
    plot: bool = True,
    force: bool = False,
):
    """
    Run ADAPT-VQE with a UCC excitation pool.

    Returns
    -------
    dict
        {
          "energy": float,
          "energies": [float],                 # per outer iteration (post inner-opt)
          "inner_energies": [[float]],         # per outer iteration
          "max_gradients": [float],            # per outer iteration (pre-append)
          "selected_operators": [{"kind":..., "wires":[...]}],
          "final_params": [float],
          "num_qubits": int,
          "config": dict,
        }
    """
    ensure_dirs()
    np.random.seed(int(seed))

    mapping_norm = str(mapping).strip().lower()

    # Hamiltonian + metadata
    (
        H,
        num_wires,
        hf_state_meta,
        symbols,
        coordinates,
        basis,
        charge,
        unit_out,
    ) = build_hamiltonian(str(molecule), mapping=mapping_norm, unit="angstrom")

    basis = str(basis).strip().lower()

    # Pool + HF (pool HF is the one consistent with qchem excitation bookkeeping)
    pool_ops, hf_state_pool = _make_ucc_pool(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        pool=str(pool),
    )

    # Prefer the pool HF if it matches; otherwise fall back to metadata HF.
    hf_state = hf_state_pool
    if len(hf_state_pool) != int(num_wires):
        hf_state = np.array(hf_state_meta, dtype=int)

    effective_noisy = is_effectively_noisy(
        bool(noisy),
        float(depolarizing_prob),
        float(amplitude_damping_prob),
        noise_model=noise_model,
    )

    if not bool(effective_noisy):
        depolarizing_prob = 0.0
        amplitude_damping_prob = 0.0

    # Config + caching
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        ansatz_desc=f"ADAPT-VQE({str(pool).strip().lower()})",
        optimizer_name=str(optimizer_name),
        stepsize=float(inner_stepsize),
        max_iterations=int(inner_steps),
        seed=int(seed),
        mapping=mapping_norm,
        noisy=bool(effective_noisy),
        depolarizing_prob=float(depolarizing_prob),
        amplitude_damping_prob=float(amplitude_damping_prob),
        molecule_label=str(molecule).strip(),
    )
    cfg["adapt_pool"] = str(pool).strip().lower()
    cfg["adapt_max_ops"] = int(max_ops)
    cfg["adapt_grad_tol"] = float(grad_tol)
    cfg["adapt_inner_steps"] = int(inner_steps)
    cfg["adapt_inner_stepsize"] = float(inner_stepsize)
    cfg["noise_model"] = (
        getattr(noise_model, "__name__", str(noise_model))
        if noise_model is not None
        else None
    )

    sig = run_signature(cfg)
    base_prefix = make_filename_prefix(
        cfg,
        noisy=bool(effective_noisy),
        seed=int(seed),
        hash_str=sig,
        algo=None,
    )
    prefix = f"{base_prefix}_adapt"

    if not force:
        record = load_run_record(prefix)
        if record is not None:
            return record["result"]

    # Device + diff method
    dev = make_device(int(num_wires), noisy=bool(effective_noisy))
    diff_method = "finite-diff" if bool(effective_noisy) else "parameter-shift"

    # ADAPT state
    selected: List[PoolOp] = []
    selected_set: set[PoolOp] = set()

    theta = np.array([], requires_grad=True)

    energies_outer: List[float] = []
    inner_energies: List[List[float]] = []
    max_gradients: List[float] = []

    # Outer loop
    for _outer in range(int(max_ops) + 1):
        # Inner optimization for current ansatz
        energy_qnode = _energy_qnode_factory(
            H=H,
            dev=dev,
            hf_state=hf_state,
            ops=selected,
            num_wires=int(num_wires),
            noisy=bool(effective_noisy),
            depolarizing_prob=float(depolarizing_prob),
            amplitude_damping_prob=float(amplitude_damping_prob),
            noise_model=noise_model,
            diff_method=diff_method,
        )

        theta, traj = _inner_optimize(
            energy_qnode=energy_qnode,
            theta_init=theta,
            optimizer_name=str(optimizer_name),
            stepsize=float(inner_stepsize),
            steps=int(inner_steps),
        )

        e_now = float(traj[-1])
        energies_outer.append(e_now)
        inner_energies.append([float(x) for x in traj])

        # Stop if we've already hit the operator budget
        if len(selected) >= int(max_ops):
            break

        # Score pool operators by gradient magnitude at theta_new=0 when appended
        best_op: Optional[PoolOp] = None
        best_grad_abs: float = -1.0

        for cand in pool_ops:
            if cand in selected_set:
                continue

            ops_plus = list(selected) + [cand]
            energy_plus = _energy_qnode_factory(
                H=H,
                dev=dev,
                hf_state=hf_state,
                ops=ops_plus,
                num_wires=int(num_wires),
                noisy=bool(effective_noisy),
                depolarizing_prob=float(depolarizing_prob),
                amplitude_damping_prob=float(amplitude_damping_prob),
                noise_model=noise_model,
                diff_method=diff_method,
            )

            theta_plus = np.concatenate([np.array(theta, dtype=float), np.array([0.0])])
            theta_plus = np.array(theta_plus, requires_grad=True)

            g = qml.grad(energy_plus)(theta_plus)
            g_last = float(np.abs(g[-1]))

            if g_last > best_grad_abs:
                best_grad_abs = g_last
                best_op = cand

        max_gradients.append(float(best_grad_abs))

        # Convergence check
        if best_op is None or float(best_grad_abs) < float(grad_tol):
            break

        # Append best operator with initial parameter 0
        selected.append(best_op)
        selected_set.add(best_op)
        theta = np.array(
            np.concatenate([np.array(theta, dtype=float), np.array([0.0])]),
            requires_grad=True,
        )

    result = {
        "energy": float(energies_outer[-1]) if energies_outer else float("nan"),
        "energies": [float(x) for x in energies_outer],
        "inner_energies": inner_energies,
        "max_gradients": [float(x) for x in max_gradients],
        "selected_operators": [
            {"kind": op.kind, "wires": [int(w) for w in op.wires]} for op in selected
        ],
        "final_params": [float(x) for x in np.array(theta, dtype=float).tolist()],
        "num_qubits": int(num_wires),
        "config": cfg,
    }

    save_run_record(prefix, {"config": cfg, "result": result})
    print(f"\n💾 Saved ADAPT-VQE run record: results/vqe/{prefix}.json\n")

    if plot:
        try:
            import matplotlib.pyplot as plt

            from common.plotting import build_filename, save_plot

            plt.figure(figsize=(8, 5))
            plt.plot(range(len(energies_outer)), energies_outer, marker="o")
            plt.xlabel("ADAPT iteration")
            plt.ylabel("Energy (Ha)")
            plt.grid(True, alpha=0.35)
            plt.title(f"{molecule} — ADAPT-VQE ({str(pool).strip().upper()})")
            plt.tight_layout()

            fname = build_filename(
                topic="adapt_conv",
                ansatz=f"ADAPT_{str(pool).strip().upper()}",
                optimizer=str(optimizer_name),
                mapping=mapping_norm,
                seed=int(seed),
                dep=float(depolarizing_prob) if bool(effective_noisy) else None,
                amp=float(amplitude_damping_prob) if bool(effective_noisy) else None,
                noise_scan=False,
                multi_seed=False,
            )
            save_plot(fname, kind="vqe", molecule=str(molecule).strip(), show=True)
        except Exception as exc:
            print(f"⚠️ ADAPT-VQE plotting failed (non-fatal): {exc}")

    return result
