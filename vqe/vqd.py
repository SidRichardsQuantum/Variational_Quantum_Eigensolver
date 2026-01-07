# vqe/vqd.py
from __future__ import annotations

import json
from typing import Callable, List, Optional

import pennylane as qml
from pennylane import numpy as np

from .engine import (
    build_ansatz,
    build_optimizer,
    make_device,
    make_energy_qnode,
    make_state_qnode,
)
from .hamiltonian import build_hamiltonian
from .io_utils import (
    RESULTS_DIR,
    ensure_dirs,
    make_filename_prefix,
    make_run_config_dict,
    run_signature,
    save_run_record,
)
from .visualize import plot_multi_state_convergence


def _state_overlap_metric(state_a, state_b, noisy: bool):
    """
    Overlap metric used in VQD.
      - noiseless: |<psi_a|psi_b>|^2
      - noisy:     Tr(rho_a rho_b)

    Notes
    -----
    - Returns an autograd-compatible scalar (do NOT cast to float here).
    - `state_a` is typically a detached reference (np.array(..., requires_grad=False)).
    - `state_b` should remain differentiable so the penalty can influence optimization.
    """
    if not noisy:
        # |<a|b>|^2 = |sum_i conj(a_i) b_i|^2
        inner = (np.conj(state_a) * state_b).sum()
        val = np.abs(inner) ** 2
        return np.clip(val, 0.0, 1.0)

    # density matrices: Tr(rho_a rho_b)
    rho_a = np.array(state_a)
    rho_b = np.array(state_b)

    # Tr(rho_a rho_b) = sum_{i,j} rho_a[i,j] * rho_b[j,i]
    val = (rho_a * np.transpose(rho_b)).sum()
    val = np.real(val)
    return np.clip(val, 0.0, 1.0)


def _beta_schedule(
    t: int,
    steps: int,
    *,
    beta_start: float,
    beta_end: float,
    ramp: str = "linear",
    hold_fraction: float = 0.0,
):
    """
    Beta schedule for deflation penalty.

    Parameters
    ----------
    t
        Current optimization step index (0-based).
    steps
        Total number of optimization steps.
    beta_start
        Initial beta value at t=0 (or during hold period).
    beta_end
        Final beta value at t=steps-1.
    ramp
        "linear" or "cosine".
    hold_fraction
        Fraction in [0,1) of steps to hold beta at beta_start before ramping.

    Returns
    -------
    autograd-friendly scalar beta(t)
    """
    if steps <= 1:
        return beta_end

    ramp = str(ramp).strip().lower()
    hold_fraction = float(hold_fraction)
    if not (0.0 <= hold_fraction < 1.0):
        raise ValueError("beta_hold_fraction must be in [0, 1).")

    hold_steps = int(np.floor(hold_fraction * steps))
    if t < hold_steps:
        return beta_start

    # Normalize progress in [0,1]
    denom = max(1, steps - 1 - hold_steps)
    x = (t - hold_steps) / denom

    if ramp == "linear":
        s = x
    elif ramp == "cosine":
        # smooth start/end: s = 0.5*(1 - cos(pi x))
        s = 0.5 * (1.0 - np.cos(np.pi * x))
    else:
        raise ValueError("beta_ramp must be 'linear' or 'cosine'.")

    return beta_start + (beta_end - beta_start) * s


def run_vqd(
    molecule: str = "H3+",
    *,
    num_states: int = 2,
    beta: float = 10.0,
    beta_start: Optional[float] = None,
    beta_ramp: str = "linear",
    beta_hold_fraction: float = 0.0,
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    steps: int = 100,
    stepsize: float = 0.4,
    seed: int = 0,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[[list[int]], None]] = None,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    plot: bool = True,
    force: bool = False,
):
    """
    Variational Quantum Deflation (VQD) for ground + excited states (k-state).

    Sequential algorithm:
      - Solve VQE for state 0
      - For state n>0 minimize:
            E(theta_n) + beta(t) * sum_{k<n} overlap(state_k, state_n)

    where overlap is:
      - noiseless: |<psi_k|psi_n>|^2
      - noisy:     Tr(rho_k rho_n)

    Beta schedule:
      - beta_end := beta
      - beta_start defaults to 0.0 if not provided
      - beta(t) ramps from beta_start -> beta_end over optimization steps
        (optionally holding at beta_start for an initial fraction).

    Noise (generalized):
      - If noisy=True, the device is default.mixed and the state QNode returns a density matrix.
      - You may specify:
          * depolarizing_prob, amplitude_damping_prob (legacy convenience)
          * noise_model(wires) applying arbitrary PennyLane noise channels
        Both are supported and can be combined.

    Returns
    -------
    dict with keys:
      - energies_per_state: list[list[float]]   (length num_states)
      - final_params:       list[list[float]]   (length num_states)
      - config:             dict
    """
    if num_states < 2:
        raise ValueError("VQD is intended for num_states >= 2")
    if steps < 1:
        raise ValueError("steps must be >= 1")

    beta_end = float(beta)
    beta0 = 0.0 if beta_start is None else float(beta_start)

    np.random.seed(seed)
    ensure_dirs()

    # 1) Hamiltonian + molecular data
    if symbols is None or coordinates is None:
        H, num_wires, symbols, coordinates, basis = build_hamiltonian(molecule)
    else:
        charge = +1 if str(molecule).upper() == "H3+" else 0
        H, num_wires = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=charge, basis=basis, unit="angstrom"
        )

    # 2) Ansatz (for QNode construction)
    ansatz_fn, _ = build_ansatz(
        ansatz_name,
        num_wires,
        seed=seed,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )

    # 3) Device + QNodes
    dev = make_device(num_wires, noisy=noisy)

    energy_qnode = make_energy_qnode(
        H,
        dev,
        ansatz_fn,
        num_wires,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        noise_model=noise_model,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )

    # Important: state-returning QNodes cannot use parameter-shift
    state_qnode = make_state_qnode(
        dev,
        ansatz_fn,
        num_wires,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        noise_model=noise_model,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        diff_method="finite-diff",
    )

    # 4) Config + caching
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        ansatz_desc=f"VQD({ansatz_name})_{num_states}states",
        optimizer_name=optimizer_name,
        stepsize=stepsize,
        max_iterations=steps,
        seed=seed,
        mapping="jordan_wigner",
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        molecule_label=molecule,
    )
    cfg["beta_end"] = beta_end
    cfg["beta_start"] = beta0
    cfg["beta_ramp"] = str(beta_ramp)
    cfg["beta_hold_fraction"] = float(beta_hold_fraction)
    cfg["num_states"] = int(num_states)

    # Noise model is not JSON-serializable; store only a lightweight identifier.
    if noise_model is not None:
        cfg["noise_model"] = getattr(noise_model, "__name__", str(noise_model))
    else:
        cfg["noise_model"] = None

    sig = run_signature(cfg)
    prefix = make_filename_prefix(cfg, noisy=noisy, seed=seed, hash_str=sig, algo="VQD")
    result_path = RESULTS_DIR / f"{prefix}.json"

    if not force and result_path.exists():
        print(f"📂 Using cached VQD result: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # 5) Storage
    energies_per_state: List[List[float]] = [[] for _ in range(num_states)]
    final_params: List[np.ndarray] = []
    reference_states: List[np.ndarray] = []  # detached references for deflation

    # 6) Solve sequentially for n=0..num_states-1
    for n in range(num_states):
        opt = build_optimizer(optimizer_name, stepsize=stepsize)

        # Init params for this state (offset seed to diversify starts)
        _, p_init = build_ansatz(
            ansatz_name,
            num_wires,
            seed=seed + n,
            symbols=symbols,
            coordinates=coordinates,
            basis=basis,
        )
        theta = np.array(p_init, requires_grad=True)

        # Ground state: pure energy objective
        if n == 0:

            def cost(th):
                return energy_qnode(th)

            for _ in range(steps):
                try:
                    theta, _ = opt.step_and_cost(cost, theta)
                except AttributeError:
                    theta = opt.step(cost, theta)

                energies_per_state[n].append(float(energy_qnode(theta)))

            final_params.append(np.array(theta, requires_grad=False))
            reference_states.append(np.array(state_qnode(theta), requires_grad=False))
            continue

        # Excited states: energy + deflation with beta ramp
        def cost_factory(t: int):
            b = _beta_schedule(
                t,
                steps,
                beta_start=beta0,
                beta_end=beta_end,
                ramp=beta_ramp,
                hold_fraction=beta_hold_fraction,
            )

            def _cost(th):
                e = energy_qnode(th)
                st = state_qnode(
                    th
                )  # keep differentiable so penalty shapes optimization
                pen = 0.0
                for prev in reference_states:
                    pen = pen + _state_overlap_metric(prev, st, noisy=noisy)
                return e + b * pen

            return _cost

        for t in range(steps):
            cost_t = cost_factory(t)
            try:
                theta, _ = opt.step_and_cost(cost_t, theta)
            except AttributeError:
                theta = opt.step(cost_t, theta)

            energies_per_state[n].append(float(energy_qnode(theta)))

        final_params.append(np.array(theta, requires_grad=False))
        reference_states.append(np.array(state_qnode(theta), requires_grad=False))

    # 7) Save
    result = {
        "energies_per_state": energies_per_state,
        "final_params": [p.tolist() for p in final_params],
        "config": cfg,
    }
    save_run_record(prefix, {"config": cfg, "result": result})
    print(f"💾 Saved VQD run to {result_path}")

    # 8) Plot
    if plot:
        try:
            plot_multi_state_convergence(
                ssvqe_or_vqd="VQD",
                energies_per_state=energies_per_state,
                molecule=molecule,
                ansatz=ansatz_name,
                optimizer=optimizer_name,
                seed=seed,
                show=True,
                save=True,
            )
        except Exception as exc:
            print(f"⚠️ VQD plotting failed (non-fatal): {exc}")

    return result
