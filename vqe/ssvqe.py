# vqe/ssvqe.py
from __future__ import annotations
import os, json, itertools
import pennylane as qml
from pennylane import numpy as np

from .hamiltonian import build_hamiltonian
from .engine import (
    make_device, build_ansatz, build_optimizer,
    make_energy_qnode, make_overlap00_fn,
)
from .io_utils import (
    ensure_dirs, make_run_config_dict, run_signature, save_run_record, IMG_DIR,
)


def run_ssvqe(
    molecule: str = "H3+",
    *,
    # algorithmic
    num_states: int = 2,                  # ground + (num_states-1) excited
    penalty_weight: float = 10.0,
    # ansatz / optimizer
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    steps: int = 100,
    stepsize: float = 0.4,
    seed: int = 0,
    # noise
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    # geometry override
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    # misc
    plot: bool = True,
    force: bool = False,
):
    """
    General SSVQE solver:
      - Works with ANY ansatz in vqe.ansatz (UCCSD, RY-CZ, StronglyEntanglingLayers, ...)
      - Works with ANY optimizer in vqe.optimizer (Adam, Momentum, ...)
      - Supports optional noise (same flags as run_vqe)

    Returns
    -------
    dict with keys:
      energies_per_state : list[list[float]]   # per-iteration energies for each state 0..k-1
      final_params       : list[array]         # final parameter arrays for each state
      config             : dict                # reproducible configuration (for caching)
    """
    assert num_states >= 2, "SSVQE requires at least two states."
    np.random.seed(seed)
    ensure_dirs()

    # --- Hamiltonian & molecule info ---
    if symbols is None or coordinates is None:
        H, num_wires, symbols, coordinates, basis = build_hamiltonian(molecule)
    else:
        H, num_wires = qml.qchem.molecular_hamiltonian(
            symbols, coordinates,
            charge=+1 if molecule.upper() == "H3+" else 0,
            basis=basis, unit="angstrom"
        )

    # --- Ansatz (agnostic) & initial params for each state ---
    ansatz_fn, init_p = build_ansatz(ansatz_name, num_wires,
                                     seed=seed, symbols=symbols, coordinates=coordinates, basis=basis)
    # Create per-state parameter sets (independent copies)
    param_sets = [np.array(init_p + 0.0, requires_grad=True) for _ in range(num_states)]

    # --- Device & QNodes ---
    dev = make_device(num_wires, noisy=noisy)
    energy_qnode = make_energy_qnode(
        H, dev, ansatz_fn, num_wires,
        noisy=noisy, depolarizing_prob=depolarizing_prob, amplitude_damping_prob=amplitude_damping_prob,
        symbols=symbols, coordinates=coordinates, basis=basis
    )
    overlap00 = make_overlap00_fn(
        dev, ansatz_fn, num_wires,
        noisy=noisy, depolarizing_prob=depolarizing_prob, amplitude_damping_prob=amplitude_damping_prob,
        symbols=symbols, coordinates=coordinates, basis=basis
    )

    # --- Config & cache signature (consistent with run_vqe) ---
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        ansatz_desc=f"SSVQE({ansatz_name}) {num_states}-state",
        optimizer_name=optimizer_name,
        stepsize=stepsize,
        max_iterations=steps,
        seed=seed,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
    )
    cfg["penalty_weight"] = float(penalty_weight)
    sig = run_signature(cfg)
    prefix = f"{molecule.replace('+','plus')}_SSVQE_{ansatz_name}_{optimizer_name}_s{seed}__{sig}"
    result_path = os.path.join("results", f"{prefix}.json")

    if not force and os.path.exists(result_path):
        print(f"📂 Using cached SSVQE result: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # --- Optimizer (agnostic) ---
    opt = build_optimizer(optimizer_name, stepsize=stepsize)

    # --- Cost: sum of state energies + pairwise overlap penalties ---
    def cost(flat_params):
        # flat_params = concatenation of all param arrays
        slices, starts = [], 0
        for p in param_sets:
            n = int(np.prod(p.shape))
            slices.append((starts, starts + n, p.shape))
            starts += n
        unpacked = []
        for (s, e, shape) in slices:
            unpacked.append(np.reshape(flat_params[s:e], shape))

        # energies
        energies = [energy_qnode(p) for p in unpacked]
        total = np.sum(energies)

        # orthogonality penalties
        for i, j in itertools.combinations(range(len(unpacked)), 2):
            total = total + penalty_weight * overlap00(unpacked[i], unpacked[j])
        return total

    # Flatten parameters for joint optimization
    flat = np.concatenate([p.ravel() for p in param_sets])
    flat = np.array(flat, requires_grad=True)

    # Track per-iteration energies for each state
    energies_per_state = [[] for _ in range(num_states)]

    for _ in range(steps):
        try:
            flat, _ = opt.step_and_cost(cost, flat)
        except AttributeError:
            flat = opt.step(cost, flat)

        # Unpack and record energies
        idx, unpacked = 0, []
        for k in range(num_states):
            size = int(np.prod(param_sets[k].shape))
            vec = flat[idx:idx + size]
            unpacked.append(np.reshape(vec, param_sets[k].shape))
            idx += size

        for k in range(num_states):
            energies_per_state[k].append(float(energy_qnode(unpacked[k])))

        # Replace working sets (keep shapes/grad)
        param_sets = [np.array(u, requires_grad=True) for u in unpacked]

    # --- Persist & plot ---
    result = {
        "energies_per_state": energies_per_state,
        "final_params": [u.tolist() for u in param_sets],  # JSON-friendly
        "config": cfg,
    }
    save_run_record(prefix, {"config": cfg, "result": result})

    if plot and num_states >= 2:
        try:
            from .visualize import plot_ssvqe_convergence_multi
            labels = [f"E{i}" for i in range(num_states)]
            plot_ssvqe_convergence_multi(molecule, energies_per_state, labels, ansatz_name=ansatz_name, optimizer_name=optimizer_name)
        except Exception as e:
            print(f"⚠️ Plotting failed (non-fatal): {e}")

    return result
