"""
vqe.ssvqe
---------
Subspace-Search Variational Quantum Eigensolver (SSVQE).

Supports:
- Multiple excited states via orthogonality penalty
- Any ansatz or optimizer defined in the vqe package
- Optional noise models (depolarizing, amplitude damping)
- Caching and reproducibility via io_utils
"""

from __future__ import annotations
import os
import json
import itertools
import pennylane as qml
from pennylane import numpy as np

from .hamiltonian import build_hamiltonian
from .engine import (
    make_device, build_ansatz, build_optimizer,
    make_energy_qnode, make_overlap00_fn,
)
from .io_utils import (
    ensure_dirs, make_run_config_dict, run_signature, save_run_record,
)


# ================================================================
# MAIN ENTRYPOINT
# ================================================================
def run_ssvqe(
    molecule: str = "H3+",
    *,
    # Algorithmic
    num_states: int = 2,                  # Ground + excited states
    penalty_weight: float = 10.0,
    # Ansatz / optimizer
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    steps: int = 100,
    stepsize: float = 0.4,
    seed: int = 0,
    # Noise
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    # Geometry
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    # Misc
    plot: bool = True,
    force: bool = False,
):
    """
    Run a Subspace-Search VQE (SSVQE) optimization to find ground and excited states.

    Args:
        molecule: Molecule label (e.g. "H2", "LiH", "H3+").
        num_states: Number of target eigenstates (≥2).
        penalty_weight: Weight for orthogonality penalty term.
        ansatz_name: Ansatz identifier from `vqe.ansatz`.
        optimizer_name: Optimizer identifier from `vqe.optimizer`.
        steps: Number of optimization iterations.
        stepsize: Optimizer step size.
        seed: Random seed for reproducibility.
        noisy: Whether to include noise models.
        depolarizing_prob: Depolarizing channel probability.
        amplitude_damping_prob: Amplitude damping probability.
        symbols, coordinates, basis: Optional molecular data override.
        plot: Whether to plot convergence results.
        force: If True, recompute even if cached result exists.

    Returns:
        dict with keys:
            - energies_per_state : list[list[float]]
            - final_params       : list[array]
            - config             : dict (run configuration)
    """
    assert num_states >= 2, "SSVQE requires at least two states."
    np.random.seed(seed)
    ensure_dirs()

    # ============================================================
    # 1. Build Hamiltonian and molecular data
    # ============================================================
    if symbols is None or coordinates is None:
        H, num_wires, symbols, coordinates, basis = build_hamiltonian(molecule)
    else:
        charge = +1 if molecule.upper() == "H3+" else 0
        H, num_wires = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=charge, basis=basis, unit="angstrom"
        )

    # ============================================================
    # 2. Build ansatz and initialize independent parameter sets
    # ============================================================
    ansatz_fn, init_params = build_ansatz(
        ansatz_name, num_wires, seed=seed,
        symbols=symbols, coordinates=coordinates, basis=basis,
    )
    param_sets = [np.array(init_params, requires_grad=True) for _ in range(num_states)]

    # ============================================================
    # 3. Build device and QNodes
    # ============================================================
    dev = make_device(num_wires, noisy=noisy)

    energy_qnode = make_energy_qnode(
        H, dev, ansatz_fn, num_wires,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        symbols=symbols, coordinates=coordinates, basis=basis,
    )

    overlap00 = make_overlap00_fn(
        dev, ansatz_fn, num_wires,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        symbols=symbols, coordinates=coordinates, basis=basis,
    )

    # ============================================================
    # 4. Build reproducible config + caching
    # ============================================================
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
        mapping="jordan_wigner",
    )
    cfg["penalty_weight"] = float(penalty_weight)
    sig = run_signature(cfg)

    prefix = f"{molecule.replace('+', 'plus')}_SSVQE_{ansatz_name}_{optimizer_name}_s{seed}__{sig}"
    result_path = os.path.join("results", f"{prefix}.json")

    if not force and os.path.exists(result_path):
        print(f"📂 Using cached SSVQE result: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # ============================================================
    # 5. Define cost function (energy + orthogonality penalty)
    # ============================================================
    opt = build_optimizer(optimizer_name, stepsize=stepsize)

    def cost(flat_params):
        """Flattened joint cost for all state parameter vectors."""
        # Split flat vector into per-state parameter arrays
        slices, start = [], 0
        for p in param_sets:
            n = int(np.prod(p.shape))
            slices.append((start, start + n, p.shape))
            start += n

        unpacked = [
            np.reshape(flat_params[s:e], shape) for (s, e, shape) in slices
        ]

        # Sum of individual energies
        total = sum(energy_qnode(p) for p in unpacked)

        # Add orthogonality penalties between states
        for i, j in itertools.combinations(range(len(unpacked)), 2):
            total += penalty_weight * overlap00(unpacked[i], unpacked[j])

        return total

    # Flatten parameters for joint optimization
    flat = np.concatenate([p.ravel() for p in param_sets])
    flat = np.array(flat, requires_grad=True)

    energies_per_state = [[] for _ in range(num_states)]

    # ============================================================
    # 6. Optimization loop
    # ============================================================
    for step in range(steps):
        try:
            flat, _ = opt.step_and_cost(cost, flat)
        except AttributeError:
            flat = opt.step(cost, flat)

        # --- Unpack parameter vector ---
        unpacked, idx = [], 0
        for k in range(num_states):
            size = int(np.prod(param_sets[k].shape))
            vec = flat[idx:idx + size]
            unpacked.append(np.reshape(vec, param_sets[k].shape))
            idx += size

        # --- Record energies ---
        for k in range(num_states):
            energies_per_state[k].append(float(energy_qnode(unpacked[k])))

        # --- Update parameter sets for next iteration ---
        param_sets = [np.array(u, requires_grad=True) for u in unpacked]

    # ============================================================
    # 7. Persist results and optionally plot
    # ============================================================
    result = {
        "energies_per_state": energies_per_state,
        "final_params": [u.tolist() for u in param_sets],
        "config": cfg,
    }

    save_run_record(prefix, {"config": cfg, "result": result})

    if plot and num_states >= 2:
        try:
            from .visualize import plot_ssvqe_convergence_multi
            labels = [f"E{i}" for i in range(num_states)]
            plot_ssvqe_convergence_multi(
                molecule, energies_per_state, labels,
                ansatz_name=ansatz_name, optimizer_name=optimizer_name,
            )
        except Exception as e:
            print(f"⚠️ Plotting failed (non-fatal): {e}")

    return result
