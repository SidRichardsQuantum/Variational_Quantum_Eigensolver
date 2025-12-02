"""
vqe.core
--------
High-level orchestration of Variational Quantum Eigensolver (VQE) workflows.

Includes:
- Main VQE runner (`run_vqe`)
- Noise studies and multi-seed averaging
- Optimizer / ansatz comparisons
- Geometry scans (bond lengths, angles)
- Fermion-to-qubit mapping comparisons
"""

from __future__ import annotations

import os
import json

import pennylane as qml
from pennylane import numpy as np

from .hamiltonian import build_hamiltonian, generate_geometry
from .visualize import (
    plot_convergence,
    plot_optimizer_comparison,
    plot_ansatz_comparison,
    plot_noise_statistics,
)
from .io_utils import (
    IMG_DIR,
    RESULTS_DIR,
    make_run_config_dict,
    make_filename_prefix,
    run_signature,
    save_run_record,
    ensure_dirs,
)
from .engine import (
    make_device,
    build_ansatz as engine_build_ansatz,
    build_optimizer as engine_build_optimizer,
    make_energy_qnode,
    make_state_qnode,
)


# ================================================================
# SHARED HELPERS
# ================================================================
def compute_fidelity(pure_state, state_or_rho):
    """
    Fidelity between a pure state |ψ⟩ and either:
        - a statevector |φ⟩
        - or a density matrix ρ

    Returns |⟨ψ|φ⟩|² or ⟨ψ|ρ|ψ⟩ respectively.
    """
    state_or_rho = np.array(state_or_rho)
    pure_state = np.array(pure_state)

    if state_or_rho.ndim == 1:
        return float(abs(np.vdot(pure_state, state_or_rho)) ** 2)
    elif state_or_rho.ndim == 2:
        return float(np.real(np.vdot(pure_state, state_or_rho @ pure_state)))

    raise ValueError("Invalid state shape for fidelity computation")


# ================================================================
# MAIN VQE EXECUTION
# ================================================================
def run_vqe(
    molecule: str = "H2",
    seed: int = 0,
    n_steps: int = 50,
    stepsize: float = 0.2,
    plot: bool = True,
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    mapping: str = "jordan_wigner",
):
    """
    Run a Variational Quantum Eigensolver (VQE) workflow end-to-end.

    The behaviour is designed to match the legacy notebooks, while using the new
    engine/ansatz modules internally.

    Parameters
    ----------
    molecule : str
        Molecular label (used when symbols/coordinates are not explicitly provided).
    seed : int
        RNG seed for parameter initialisation and any stochastic components.
    n_steps : int
        Number of optimisation steps.
    stepsize : float
        Optimizer learning rate.
    plot : bool
        If True, plot the convergence curve.
    ansatz_name : str
        Name of the ansatz from vqe.ansatz.ANSATZES.
    optimizer_name : str
        Name of the classical optimizer.
    noisy : bool
        Whether to include depolarizing / amplitude-damping noise.
    depolarizing_prob : float
        Depolarizing channel probability (per qubit).
    amplitude_damping_prob : float
        Amplitude damping probability (per qubit).
    force : bool
        If True, ignore cached results and rerun optimisation.
    symbols, coordinates :
        Optional direct molecular specification; if provided, qchem is used directly.
    basis : str
        Basis set string (e.g. "sto-3g", "STO-3G").
    mapping : str
        Fermion-to-qubit mapping label ("jordan_wigner", etc.).

    Returns
    -------
    dict
        {
            "energy": float,
            "energies": [float, ...],
            "steps": int,
            "final_state_real": [...],
            "final_state_imag": [...],
            "num_qubits": int,
        }
    """
    ensure_dirs()
    np.random.seed(seed)

    # --- Hamiltonian & molecular data ---
    if symbols is not None and coordinates is not None:
        # Direct molecular override (used in geometry scans, etc.)
        # Normalise basis for qchem
        basis = basis.lower()
        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=0, basis=basis, unit="angstrom"
        )
    else:
        # Use shared build_hamiltonian (which already embeds mapping logic)
        H, qubits, symbols, coordinates, basis = build_hamiltonian(
            molecule, mapping=mapping
        )
        # Normalise basis string consistently
        basis = basis.lower()

    # --- Configuration & caching ---
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        ansatz_desc=ansatz_name,
        optimizer_name=optimizer_name,
        stepsize=stepsize,
        max_iterations=n_steps,
        seed=seed,
        mapping=mapping,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        molecule_label=molecule,
    )

    sig = run_signature(cfg)
    prefix = make_filename_prefix(
        cfg,
        noisy=noisy,
        seed=seed,
        hash_str=sig,
        ssvqe=False,
    )
    result_path = os.path.join(RESULTS_DIR, f"{prefix}.json")

    if not force and os.path.exists(result_path):
        print(f"\n📂 Found cached result: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # --- Device, ansatz, optim, QNodes ---
    dev = make_device(qubits, noisy=noisy)
    ansatz_fn, params = engine_build_ansatz(
        ansatz_name,
        qubits,
        seed=seed,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )
    energy_qnode = make_energy_qnode(
        H,
        dev,
        ansatz_fn,
        qubits,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )
    state_qnode = make_state_qnode(
        dev,
        ansatz_fn,
        qubits,
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )
    opt = engine_build_optimizer(optimizer_name, stepsize=stepsize)

    # --- Optimization loop ---
    params = np.array(params, requires_grad=True)
    energies = [float(energy_qnode(params))]

    for step in range(n_steps):
        try:
            # Use cost returned by step_and_cost to avoid extra QNode calls
            params, cost = opt.step_and_cost(energy_qnode, params)
            e = float(cost)
        except AttributeError:
            # Optimizers without step_and_cost
            params = opt.step(energy_qnode, params)
            e = float(energy_qnode(params))

        energies.append(e)
        print(f"Step {step + 1:02d}/{n_steps}: E = {e:.6f} Ha")

    final_energy = float(energies[-1])
    final_state = state_qnode(params)

    # --- Optional plot ---
    if plot:
        plot_convergence(
            energies,
            molecule,
            optimizer=optimizer_name,
            ansatz=ansatz_name,
        )

    # --- Save ---
    result = {
        "energy": final_energy,
        "energies": [float(e) for e in energies],
        "steps": n_steps,
        "final_state_real": np.real(final_state).tolist(),
        "final_state_imag": np.imag(final_state).tolist(),
        "num_qubits": qubits,
    }

    record = {"config": cfg, "result": result}
    save_run_record(prefix, record)
    print(f"\n💾 Saved run record to {result_path}\n")

    return result


# ================================================================
# NOISE SWEEP (SINGLE-SEED)
# ================================================================
def run_vqe_noise_sweep(
    molecule="H2",
    ansatz_name="RY-CZ",
    optimizer_name="Adam",
    steps=30,
    depolarizing_probs=None,
    amplitude_damping_probs=None,
    force=False,
    mapping: str = "jordan_wigner",
    show: bool = True,
):
    """
    Simple single-seed noise sweep:
    - Compute a noiseless reference
    - Sweep over noise probabilities and record ΔE and fidelity

    Parameters
    ----------
    show : bool
        Whether to display the generated plot (via matplotlib).
    """
    depolarizing_probs = (
        np.arange(0.0, 0.11, 0.02)
        if depolarizing_probs is None
        else np.asarray(depolarizing_probs)
    )
    amplitude_damping_probs = (
        np.zeros_like(depolarizing_probs)
        if amplitude_damping_probs is None
        else np.asarray(amplitude_damping_probs)
    )

    # --- Reference run (noiseless) ---
    ref = run_vqe(
        molecule=molecule,
        n_steps=steps,
        stepsize=0.2,
        plot=False,
        ansatz_name=ansatz_name,
        optimizer_name=optimizer_name,
        noisy=False,
        mapping=mapping,
        force=force,
    )
    reference_energy = ref["energy"]
    pure_state = np.array(ref["final_state_real"]) + 1j * np.array(
        ref["final_state_imag"]
    )

    energy_means, energy_stds = [], []
    fidelity_means, fidelity_stds = [], []

    # --- Sweep noise ---
    for p_dep, p_amp in zip(depolarizing_probs, amplitude_damping_probs):
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            stepsize=0.2,
            plot=False,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            noisy=True,
            depolarizing_prob=float(p_dep),
            amplitude_damping_prob=float(p_amp),
            mapping=mapping,
            force=force,
        )

        energy = res["energy"]
        state = np.array(res["final_state_real"]) + 1j * np.array(
            res["final_state_imag"]
        )

        dE = energy - reference_energy
        F = compute_fidelity(pure_state, state)

        energy_means.append(dE)
        energy_stds.append(0.0)  # single seed
        fidelity_means.append(F)
        fidelity_stds.append(0.0)

    # Decide noise label for plots
    if np.allclose(amplitude_damping_probs, 0.0):
        noise_type = "Depolarizing"
        noise_levels = depolarizing_probs
    elif np.allclose(depolarizing_probs, 0.0):
        noise_type = "Amplitude"
        noise_levels = amplitude_damping_probs
    else:
        noise_type = "Combined"
        noise_levels = depolarizing_probs  # x-axis label; both are meaningful

    plot_noise_statistics(
        molecule,
        noise_levels,
        energy_means,
        energy_stds,
        fidelity_means,
        fidelity_stds,
        optimizer_name=optimizer_name,
        ansatz_name=ansatz_name,
        noise_type=noise_type,
        show=show,
    )

    print(f"\n✅ Noise sweep complete for {molecule} ({ansatz_name}, {optimizer_name})")


# ================================================================
# OPTIMIZER COMPARISON
# ================================================================
def run_vqe_optimizer_comparison(
    molecule="H2",
    ansatz_name="RY-CZ",
    optimizers=None,
    steps=50,
    stepsize=0.2,
    noisy=True,
    depolarizing_prob=0.05,
    amplitude_damping_prob=0.05,
    force=False,
    mapping: str = "jordan_wigner",
    show: bool = True,
    seed=0,
):
    """
    Compare different classical optimizers on the same VQE instance.

    Parameters
    ----------
    show : bool
        Whether to display the generated plot.

    Returns
    -------
    dict
        {
            "energies": {opt_name: [E0, E1, ...], ...},
            "final_energies": {opt_name: E_final, ...}
        }
    """
    import matplotlib.pyplot as plt
    from vqe_qpe_common.plotting import build_filename, save_plot

    optimizers = optimizers or ["Adam", "GradientDescent", "Momentum"]
    results = {}
    final_vals = {}

    # --- Run each optimizer ---
    for opt_name in optimizers:
        print(f"\n⚙️ Running optimizer: {opt_name}")

        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            stepsize=stepsize,
            plot=False,
            ansatz_name=ansatz_name,
            optimizer_name=opt_name,
            noisy=noisy,
            depolarizing_prob=depolarizing_prob,
            amplitude_damping_prob=amplitude_damping_prob,
            mapping=mapping,
            force=force,
            seed=seed,
        )

        results[opt_name] = res["energies"]
        final_vals[opt_name] = res["energy"]

    # --- Plot comparison ---
    plt.figure(figsize=(8, 5))
    min_len = min(len(v) for v in results.values())

    for opt, energies in results.items():
        plt.plot(range(min_len), energies[:min_len], label=opt)

    plt.title(f"{molecule} – Optimizer Comparison ({ansatz_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # Show first (for notebooks), then save (save_plot will close the figure)
    if show:
        plt.show()

    fname = build_filename(
        molecule=molecule,
        topic="optimizer_comparison",
        extras={"ans": ansatz_name},
    )
    save_plot(fname)

    return {
        "energies": results,
        "final_energies": final_vals,
    }


# ================================================================
# ANSATZ COMPARISON
# ================================================================
def run_vqe_ansatz_comparison(
    molecule="H2",
    optimizer_name="Adam",
    ansatzes=None,
    steps=50,
    stepsize=0.2,
    noisy=True,
    depolarizing_prob=0.05,
    amplitude_damping_prob=0.05,
    force=False,
    mapping: str = "jordan_wigner",
    show: bool = True,
):
    """
    Compare different ansatz families on the same molecule / optimizer.

    Parameters
    ----------
    show : bool
        Whether to display the generated plot.

    Returns
    -------
    dict
        {
            "energies": {ansatz_name: [E0, E1, ...], ...},
            "final_energies": {ansatz_name: E_final, ...}
        }
    """
    ansatzes = ansatzes or ["RY-CZ", "Minimal", "TwoQubit-RY-CNOT"]
    results = {}

    for ans_name in ansatzes:
        print(f"\n🔹 Running ansatz: {ans_name}")
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            stepsize=stepsize,
            plot=False,
            ansatz_name=ans_name,
            optimizer_name=optimizer_name,
            noisy=noisy,
            depolarizing_prob=depolarizing_prob,
            amplitude_damping_prob=amplitude_damping_prob,
            mapping=mapping,
            force=force,
        )
        results[ans_name] = res["energies"]

    plot_ansatz_comparison(molecule, results, optimizer=optimizer_name, show=show)
    print(f"\n✅ Ansatz comparison complete for {molecule} ({optimizer_name})")

    final_energies = {name: energies[-1] for name, energies in results.items()}
    return {
        "energies": results,
        "final_energies": final_energies,
    }


# ================================================================
# MULTI-SEED NOISE STUDIES
# ================================================================
def run_vqe_multi_seed_noise(
    molecule="H2",
    ansatz_name="RY-CZ",
    optimizer_name="Adam",
    steps=30,
    stepsize=0.2,
    seeds=None,
    noise_type="depolarizing",
    depolarizing_probs=None,
    amplitude_damping_probs=None,
    force=False,
    mapping: str = "jordan_wigner",
    show: bool = True,
):
    """
    Multi-seed noise statistics for a given molecule and ansatz.
    """
    # ---------- FIXED HANDLING FOR NUMPY ARRAYS ----------
    if seeds is None:
        seeds = np.arange(0, 5)

    if depolarizing_probs is None:
        depolarizing_probs = np.arange(0.0, 0.11, 0.02)

    if amplitude_damping_probs is None:
        amplitude_damping_probs = np.zeros_like(depolarizing_probs)

    # ---------- NOISE TYPE HANDLING ----------
    if noise_type == "depolarizing":
        amplitude_damping_probs = [0.0] * len(depolarizing_probs)

    elif noise_type == "amplitude":
        amplitude_damping_probs = depolarizing_probs
        depolarizing_probs = [0.0] * len(amplitude_damping_probs)

    elif noise_type == "combined":
        amplitude_damping_probs = depolarizing_probs.copy()

    else:
        raise ValueError(f"Unknown noise type '{noise_type}'")

    # --- Reference (noiseless) ---
    print("\n🔹 Computing noiseless reference runs...")
    ref_energies, ref_states = [], []
    for s in seeds:
        np.random.seed(int(s))
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            stepsize=stepsize,
            plot=False,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            noisy=False,
            mapping=mapping,
            force=force,
            seed=int(s),
        )
        ref_energies.append(res["energy"])
        state = np.array(res["final_state_real"]) + 1j * np.array(
            res["final_state_imag"]
        )
        ref_states.append(state)

    reference_energy = float(np.mean(ref_energies))
    reference_state = ref_states[0] / np.linalg.norm(ref_states[0])
    print(f"Reference mean energy = {reference_energy:.6f} Ha")

    # --- Noisy sweeps ---
    energy_means, energy_stds = [], []
    fidelity_means, fidelity_stds = [], []

    for p_dep, p_amp in zip(depolarizing_probs, amplitude_damping_probs):
        noisy_energies, fidelities = [], []
        for s in seeds:
            np.random.seed(int(s))
            res = run_vqe(
                molecule=molecule,
                n_steps=steps,
                stepsize=stepsize,
                plot=False,
                ansatz_name=ansatz_name,
                optimizer_name=optimizer_name,
                noisy=True,
                depolarizing_prob=float(p_dep),
                amplitude_damping_prob=float(p_amp),
                mapping=mapping,
                force=force,
                seed=int(s),
            )
            noisy_energies.append(res["energy"])
            state = np.array(res["final_state_real"]) + 1j * np.array(
                res["final_state_imag"]
            )
            state = state / np.linalg.norm(state)
            fidelities.append(compute_fidelity(reference_state, state))

        noisy_energies = np.array(noisy_energies)
        dE = noisy_energies - reference_energy

        energy_means.append(float(np.mean(dE)))
        energy_stds.append(float(np.std(dE)))
        fidelity_means.append(float(np.mean(fidelities)))
        fidelity_stds.append(float(np.std(fidelities)))

        print(
            f"Noise p_dep={float(p_dep):.2f}, p_amp={float(p_amp):.2f}: "
            f"ΔE={energy_means[-1]:.6f} ± {energy_stds[-1]:.6f}, "
            f"⟨F⟩={fidelity_means[-1]:.4f}"
        )

    noise_levels = (
        amplitude_damping_probs if noise_type == "amplitude" else depolarizing_probs
    )

    plot_noise_statistics(
        molecule,
        noise_levels,
        energy_means,
        energy_stds,
        fidelity_means,
        fidelity_stds,
        optimizer_name=optimizer_name,
        ansatz_name=ansatz_name,
        noise_type=noise_type.capitalize(),
        show=show,
    )

    print(f"\n✅ Multi-seed noise study complete for {molecule}")


# ================================================================
# GEOMETRY SCAN
# ================================================================
def run_vqe_geometry_scan(
    molecule="H2_BOND",
    param_name="bond",
    param_values=None,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=30,
    stepsize=0.2,
    seeds=None,
    force=False,
    mapping: str = "jordan_wigner",
    show: bool = True,
):
    """
    Geometry scan using run_vqe + generate_geometry, mirroring the H₂O and LiH notebooks.

    Parameters
    ----------
    show : bool
        Whether to display the generated plot.

    Returns
    -------
    list of tuples
        [(param_value, mean_E, std_E), ...]
    """
    from vqe_qpe_common.plotting import (
        build_filename,
        save_plot,
        format_molecule_name,
    )
    import matplotlib.pyplot as plt

    if param_values is None:
        raise ValueError("param_values must be specified")

    seeds = seeds or [0]
    results = []

    for val in param_values:
        print(f"\n⚙️ Geometry: {param_name} = {val:.3f}")
        symbols, coordinates = generate_geometry(molecule, val)

        energies_for_val = []
        for s in seeds:
            np.random.seed(int(s))
            res = run_vqe(
                molecule=molecule,
                n_steps=steps,
                stepsize=stepsize,
                ansatz_name=ansatz_name,
                optimizer_name=optimizer_name,
                symbols=symbols,
                coordinates=coordinates,
                noisy=False,
                plot=False,
                seed=int(s),
                force=force,
                mapping=mapping,
            )
            energies_for_val.append(res["energy"])

        mean_E = float(np.mean(energies_for_val))
        std_E = float(np.std(energies_for_val))
        results.append((val, mean_E, std_E))
        print(f"  → Mean E = {mean_E:.6f} ± {std_E:.6f} Ha")

    # --- Plot ---
    params, means, stds = zip(*results)

    plt.errorbar(params, means, yerr=stds, fmt="o-", capsize=4)
    plt.xlabel(f"{param_name.capitalize()} (Å or °)")
    plt.ylabel("Ground-State Energy (Ha)")
    plt.title(
        f"{molecule} Energy vs {param_name.capitalize()} ({ansatz_name}, {optimizer_name})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    mol_norm = format_molecule_name(molecule)
    fname = build_filename(
        molecule=mol_norm,
        topic="vqe_geometry_scan",
        extras={
            "ans": ansatz_name,
            "opt": optimizer_name,
            "param": param_name,
        },
    )
    save_plot(fname)

    min_idx = int(np.argmin(means))
    print(
        f"Minimum energy: {means[min_idx]:.6f} ± {stds[min_idx]:.6f} "
        f"at {param_name}={params[min_idx]:.3f}"
    )

    return results


# ================================================================
# MAPPING COMPARISON
# ================================================================
def run_vqe_mapping_comparison(
    molecule="H2",
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    mappings=None,
    steps=50,
    stepsize=0.2,
    noisy=False,
    depolarizing_prob=0.0,
    amplitude_damping_prob=0.0,
    force=False,
    show=True,
    mapping_kwargs=None,
):
    """
    Compare different fermion-to-qubit mappings by:

    - Building qubit Hamiltonians via build_hamiltonian
    - Running VQE (re-using caching) via run_vqe for each mapping
    - Plotting energy convergence curves and printing summary

    Parameters
    ----------
    show : bool
        Whether to display the generated plot.

    Returns
    -------
    dict
        {
            mapping_name: {
                "final_energy": float,
                "energies": [...],
                "num_qubits": int,
                "num_terms": int or None,
            },
            ...
        }
    """
    import matplotlib.pyplot as plt
    from vqe_qpe_common.plotting import build_filename, save_plot

    mappings = mappings or ["jordan_wigner", "bravyi_kitaev", "parity"]
    results = {}

    print(f"\n🔍 Comparing mappings for {molecule} ({ansatz_name}, {optimizer_name})")

    for mapping in mappings:
        print(f"\n⚙️ Running mapping: {mapping}")

        # Build Hamiltonian once to inspect complexity
        H, qubits, symbols, coordinates, basis = build_hamiltonian(
            molecule, mapping=mapping
        )
        basis = basis.lower()

        try:
            num_terms = len(H.ops)
        except AttributeError:
            try:
                num_terms = len(H.terms()[0]) if callable(H.terms) else len(H.data)
            except Exception:
                num_terms = (
                    len(getattr(H, "data", [])) if hasattr(H, "data") else None
                )

        # Run VQE using the high-level entrypoint (handles ansatz + noise plumbing)
        res = run_vqe(
            molecule=molecule,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            n_steps=steps,
            stepsize=stepsize,
            noisy=noisy,
            depolarizing_prob=depolarizing_prob,
            amplitude_damping_prob=amplitude_damping_prob,
            mapping=mapping,
            force=force,
            plot=False,
        )

        results[mapping] = {
            "final_energy": res["energy"],
            "energies": res["energies"],
            "num_qubits": qubits,
            "num_terms": num_terms,
        }

    # --- Plot mappings ---
    plt.figure(figsize=(8, 5))
    for mapping in mappings:
        data = results[mapping]
        label = mapping.replace("_", "-").title()
        plt.plot(
            range(len(data["energies"])),
            data["energies"],
            label=label,
            linewidth=2,
            alpha=0.9,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE: Energy Convergence by Mapping ({ansatz_name})")
    plt.legend(frameon=False, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=2)

    if show:
        plt.show()

    fname = build_filename(
        molecule=molecule,
        topic="mapping_comparison",
        extras={"ansatz": ansatz_name, "opt": optimizer_name},
    )
    save_plot(fname)

    print(
        f"\n📉 Saved mapping comparison plot to {IMG_DIR}/{fname}\nResults Summary:"
    )
    for mapping, data in results.items():
        print(
            f"  {mapping:15s} → E = {data['final_energy']:.8f} Ha, "
            f"Qubits = {data['num_qubits']}, Terms = {data['num_terms']}"
        )

    return results
