"""
vqe.ssvqe
---------
Subspace-Search Variational Quantum Eigensolver (SSVQE).

Features:
    - Ground and excited states via a shared variational ansatz
    - Orthogonality enforced with |⟨ψ_i|ψ_j⟩|² penalties
    - Any ansatz and optimizer defined in the vqe package
    - Optional depolarizing / amplitude-damping noise
    - Caching and reproducibility via vqe.io_utils
"""

from __future__ import annotations

import itertools
import json

import pennylane as qml
from pennylane import numpy as np

from .engine import (
    build_ansatz,
    build_optimizer,
    make_device,
    make_energy_qnode,
    make_overlap00_fn,
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
from .visualize import plot_ssvqe_convergence_multi


# ================================================================
# MAIN ENTRYPOINT
# ================================================================
def run_ssvqe(
    molecule: str = "H3+",
    *,
    num_states: int = 2,
    penalty_weight: float = 10.0,
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    steps: int = 100,
    stepsize: float = 0.4,
    seed: int = 0,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    plot: bool = True,
    force: bool = False,
):
    """
    Run a Subspace-Search VQE (SSVQE) optimization to obtain ground and
    excited states of a molecular Hamiltonian.

    Args:
        molecule: Molecule label (e.g. "H2", "LiH", "H3+").
        num_states: Number of eigenstates to target (>= 2).
        penalty_weight: Weight on the orthogonality penalty term.
        ansatz_name: Name of the ansatz from `vqe.ansatz`.
        optimizer_name: Name of optimizer from `vqe.optimizer`.
        steps: Number of optimization iterations.
        stepsize: Optimizer step size.
        seed: Random seed for reproducibility.
        noisy: If True, use a mixed-state simulator and insert noise.
        depolarizing_prob: Depolarizing noise probability per wire.
        amplitude_damping_prob: Amplitude damping probability per wire.
        symbols, coordinates, basis: Optional explicit molecular data.
        plot: Whether to plot E0 / E1 convergence (if num_states >= 2).
        force: If True, ignore cached results and recompute.

    Returns:
        dict with keys:
            - "energies_per_state": list[list[float]]
            - "final_params":       list[list[float]]
            - "config":             dict
    """
    assert num_states >= 2, "SSVQE requires at least two target states."

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
    # 2. Ansatz and different initial parameters
    # ============================================================

    ansatz_fn, p0 = build_ansatz(
        ansatz_name,
        num_wires,
        seed=seed,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )

    param_sets = [np.array(p0, requires_grad=True)]
    for k in range(1, num_states):
        _, pk = build_ansatz(
            ansatz_name,
            num_wires,
            seed=seed + k,
            symbols=symbols,
            coordinates=coordinates,
            basis=basis,
        )
        param_sets.append(np.array(pk, requires_grad=True))

    # ============================================================
    # 3. Device and QNodes
    # ============================================================
    dev = make_device(num_wires, noisy=noisy)

    energy_qnode = make_energy_qnode(
        H,
        dev,
        ansatz_fn,
        num_wires,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )

    overlap00 = make_overlap00_fn(
        dev,
        ansatz_fn,
        num_wires,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )

    # ============================================================
    # 4. Config + caching
    # ============================================================
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        ansatz_desc=f"SSVQE({ansatz_name})_{num_states}states",
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
    cfg["penalty_weight"] = float(penalty_weight)
    cfg["num_states"] = int(num_states)

    sig = run_signature(cfg)
    prefix = make_filename_prefix(
        cfg,
        noisy=noisy,
        seed=seed,
        hash_str=sig,
        ssvqe=True,
    )
    result_path = RESULTS_DIR / f"{prefix}.json"

    if not force and result_path.exists():
        print(f"📂 Using cached SSVQE result: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # ============================================================
    # 5. Cost function: total energy + orthogonality penalties
    # ============================================================
    opt = build_optimizer(optimizer_name, stepsize=stepsize)

    def _unpack_flat(flat, templates):
        """Unpack a flat parameter vector into a list of arrays matching templates."""
        arrays = []
        idx = 0
        for tmpl in templates:
            size = int(np.prod(tmpl.shape))
            vec = flat[idx : idx + size]
            arrays.append(np.reshape(vec, tmpl.shape))
            idx += size
        return arrays

    def cost(flat_params):
        # Split into per-state parameter arrays
        unpacked = _unpack_flat(flat_params, param_sets)

        # Sum of individual energies
        total = sum(energy_qnode(p) for p in unpacked)

        # Add orthogonality penalties between all distinct pairs
        for i, j in itertools.combinations(range(len(unpacked)), 2):
            total = total + penalty_weight * overlap00(unpacked[i], unpacked[j])

        return total

    # Flatten initial parameters for joint optimization
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

        # Unpack back into per-state parameter arrays
        unpacked = _unpack_flat(flat, param_sets)

        # Record energies for each state
        for k in range(num_states):
            energies_per_state[k].append(float(energy_qnode(unpacked[k])))

        # Update param_sets for the next iteration (keep grad enabled)
        param_sets = [np.array(u, requires_grad=True) for u in unpacked]

    # ============================================================
    # 7. Package and save results
    # ============================================================
    result = {
        "energies_per_state": energies_per_state,
        "final_params": [u.tolist() for u in param_sets],
        "config": cfg,
    }

    save_run_record(prefix, {"config": cfg, "result": result})
    print(f"💾 Saved SSVQE run to {result_path}")

    # ============================================================
    # 8. Optional plotting (E0 / E1 only, if available)
    # ============================================================
    if plot and num_states >= 2:
        try:
            E0 = energies_per_state[0]
            E1 = energies_per_state[1]
            plot_ssvqe_convergence_multi(
                molecule=molecule,
                ansatz=ansatz_name,
                optimizer_name=optimizer_name,
                E0_list=E0,
                E1_list=E1,
                show=True,
                save=True,
            )

        except Exception as exc:
            print(f"⚠️ SSVQE plotting failed (non-fatal): {exc}")

    return result
