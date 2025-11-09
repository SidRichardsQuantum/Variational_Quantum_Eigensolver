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

import os
import json
import pennylane as qml
from pennylane import numpy as np

from .hamiltonian import build_hamiltonian, generate_geometry
from .ansatz import get_ansatz, init_params
from .optimizer import minimize_energy
from .visualize import (
    plot_convergence,
    plot_noise_sweep,
    plot_optimizer_comparison,
    plot_ansatz_comparison,
)
from .io_utils import (
    IMG_DIR,
    RESULTS_DIR,
    make_run_config_dict,
    run_signature,
    save_run_record,
    ensure_dirs,
)

ensure_dirs()


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

    Args:
        molecule: Molecule name or identifier (e.g., "H2", "LiH").
        seed: Random seed for reproducibility.
        n_steps: Number of optimization steps.
        stepsize: Optimizer learning rate.
        plot: Whether to plot convergence curve.
        ansatz_name: Name of ansatz (e.g., "UCCSD", "RY-CZ").
        optimizer_name: Optimizer to use ("Adam", "Momentum", etc.).
        noisy: Whether to enable noise.
        depolarizing_prob, amplitude_damping_prob: Noise probabilities.
        force: Force recomputation even if cached result exists.
        symbols, coordinates, basis: Optional geometry override.
        mapping: Fermion-to-qubit mapping scheme.

    Returns:
        dict: {
            "energy": final ground-state energy,
            "energies": list of energies per step,
            "final_state_real": list,
            "final_state_imag": list,
        }
    """
    ensure_dirs()
    np.random.seed(seed)

    # --- Build or reuse molecular Hamiltonian ---
    if symbols is not None and coordinates is not None:
        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=0, basis=basis, unit="angstrom"
        )
    else:
        H, qubits, symbols, coordinates, basis = build_hamiltonian(molecule, mapping=mapping)

    ansatz_fn = get_ansatz(ansatz_name)

    # --- Configuration and caching ---
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
    )

    sig = run_signature(cfg)
    prefix = f"{molecule}_{optimizer_name}_s{seed}__{sig}"
    result_path = os.path.join(RESULTS_DIR, f"{prefix}.json")

    if not force and os.path.exists(result_path):
        print(f"\n📂 Found cached result for this configuration: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # --- Device and QNodes ---
    dev_name = "default.mixed" if noisy else "default.qubit"
    diff_method = "finite-diff" if noisy else "parameter-shift"
    dev = qml.device(dev_name, wires=qubits)

    import inspect

    def call_ansatz(ansatz_fn, params, wires, symbols, coordinates, basis):
        """Call ansatz with molecule info only if required by its signature."""
        sig = inspect.signature(ansatz_fn).parameters
        kwargs = {}
        if "symbols" in sig:
            kwargs["symbols"] = symbols
        if "coordinates" in sig:
            kwargs["coordinates"] = coordinates
        if "basis" in sig:
            kwargs["basis"] = basis
        return ansatz_fn(params, wires=wires, **kwargs)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params):
        call_ansatz(ansatz_fn, params, range(qubits), symbols, coordinates, basis)
        if noisy:
            for w in range(qubits):
                if depolarizing_prob > 0:
                    qml.DepolarizingChannel(depolarizing_prob, wires=w)
                if amplitude_damping_prob > 0:
                    qml.AmplitudeDamping(amplitude_damping_prob, wires=w)
        return qml.expval(H)

    @qml.qnode(dev, diff_method=diff_method)
    def get_state(params):
        call_ansatz(ansatz_fn, params, range(qubits), symbols, coordinates, basis)
        if noisy:
            for w in range(qubits):
                if depolarizing_prob > 0:
                    qml.DepolarizingChannel(depolarizing_prob, wires=w)
                if amplitude_damping_prob > 0:
                    qml.AmplitudeDamping(amplitude_damping_prob, wires=w)
        return qml.state()

    # --- Initialization and optimization ---
    params = init_params(ansatz_name, num_wires=qubits, symbols=symbols, coordinates=coordinates, basis=basis)
    params, energies = minimize_energy(circuit, params, optimizer_name, steps=n_steps, stepsize=stepsize)
    final_energy = float(energies[-1])
    final_state = get_state(params)

    # --- Plot convergence ---
    if plot:
        plot_convergence(energies, molecule)

    # --- Save record ---
    record = {
        "config": cfg,
        "result": {
            "energy": final_energy,
            "energies": [float(e) for e in energies],
            "steps": n_steps,
            "final_state_real": np.real(final_state).tolist(),
            "final_state_imag": np.imag(final_state).tolist(),
        },
    }
    save_run_record(prefix, record)
    print(f"\n💾 Saved run record to {result_path}\n")

    return record["result"]


# ================================================================
# NOISE STUDIES
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
):
    """Run VQE across multiple noise levels and plot convergence curves."""
    depolarizing_probs = depolarizing_probs or np.arange(0.0, 0.11, 0.02)
    amplitude_damping_probs = amplitude_damping_probs or [0.0] * len(depolarizing_probs)

    results = []
    for p_dep, p_amp in zip(depolarizing_probs, amplitude_damping_probs):
        print(f"\n🔸 Running noise level p_dep={p_dep:.2f}, p_amp={p_amp:.2f}")
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            plot=False,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            noisy=True,
            depolarizing_prob=p_dep,
            amplitude_damping_prob=p_amp,
            force=force,
            mapping=mapping,
        )
        results.append((p_dep, res["energies"]))

    plot_noise_sweep(molecule, results, optimizer=optimizer_name, ansatz=ansatz_name)
    print(f"\n✅ Completed noise sweep for {molecule}")


# ================================================================
# OPTIMIZER & ANSATZ COMPARISONS
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
):
    """Compare multiple optimizers on the same molecule and ansatz."""
    optimizers = optimizers or ["Adam", "GradientDescent", "Momentum"]
    results = {}

    for opt_name in optimizers:
        print(f"\n⚙️ Running optimizer: {opt_name}")
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            stepsize=stepsize,
            ansatz_name=ansatz_name,
            optimizer_name=opt_name,
            noisy=noisy,
            depolarizing_prob=depolarizing_prob,
            amplitude_damping_prob=amplitude_damping_prob,
            plot=False,
            force=force,
            mapping=mapping,
        )
        results[opt_name] = res["energies"]

    plot_optimizer_comparison(molecule, results, ansatz=ansatz_name)
    print(f"\n✅ Optimizer comparison complete for {molecule} ({ansatz_name})")


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
):
    """Compare multiple ansatz circuits under a fixed optimizer."""
    ansatzes = ansatzes or ["RY-CZ", "Minimal", "TwoQubit-RY-CNOT"]
    results = {}

    for ans_name in ansatzes:
        print(f"\n🔹 Running ansatz: {ans_name}")
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            stepsize=stepsize,
            ansatz_name=ans_name,
            optimizer_name=optimizer_name,
            noisy=noisy,
            depolarizing_prob=depolarizing_prob,
            amplitude_damping_prob=amplitude_damping_prob,
            plot=False,
            force=force,
            mapping=mapping,
        )
        results[ans_name] = res["energies"]

    plot_ansatz_comparison(molecule, results, optimizer=optimizer_name)
    print(f"\n✅ Ansatz comparison complete for {molecule} ({optimizer_name})")


# ================================================================
# FIDELITY & MULTI-SEED NOISE STUDIES
# ================================================================
def compute_fidelity(pure_state, state_or_rho):
    """Compute fidelity between a pure state and a (pure or mixed) target."""
    state_or_rho = np.array(state_or_rho)
    pure_state = np.array(pure_state)
    if state_or_rho.ndim == 1:
        return float(abs(np.vdot(pure_state, state_or_rho)) ** 2)
    elif state_or_rho.ndim == 2:
        return float(np.real(np.vdot(pure_state, state_or_rho @ pure_state)))
    raise ValueError("Invalid state shape for fidelity computation")


def run_vqe_multi_seed_noise(
    molecule="H2",
    ansatz_name="RY-CZ",
    optimizer_name="Adam",
    steps=30,
    stepsize=0.2,
    seeds=None,
    noise_type="depolarizing",  # "depolarizing" | "amplitude" | "combined"
    depolarizing_probs=None,
    amplitude_damping_probs=None,
    force=False,
    mapping: str = "jordan_wigner",
):
    """
    Run VQE with multiple random seeds and noise levels, collecting energy error
    and fidelity statistics relative to noiseless references.
    """
    from .visualize import plot_noise_statistics

    seeds = seeds or np.arange(0, 5)
    if depolarizing_probs is None:
        depolarizing_probs = np.arange(0.0, 0.11, 0.02)

    # --- Configure noise pairings ---
    if noise_type == "depolarizing":
        amplitude_damping_probs = [0.0] * len(depolarizing_probs)
    elif noise_type == "amplitude":
        amplitude_damping_probs = depolarizing_probs
        depolarizing_probs = [0.0] * len(amplitude_damping_probs)
    elif noise_type == "combined":
        amplitude_damping_probs = depolarizing_probs.copy()
    else:
        raise ValueError(f"Unknown noise type '{noise_type}'")

    # ============================================================
    # 1. Compute noiseless references
    # ============================================================
    print("\n🔹 Computing noiseless reference runs...")
    ref_energies, ref_states = [], []
    for seed in seeds:
        np.random.seed(int(seed))
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            stepsize=stepsize,
            plot=False,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            noisy=False,
            force=force,
            mapping=mapping,
        )
        ref_energies.append(res["energy"])
        state = np.array(res["final_state_real"]) + 1j * np.array(res["final_state_imag"])
        ref_states.append(state)

    reference_energy = float(np.mean(ref_energies))
    reference_state = ref_states[0] / np.linalg.norm(ref_states[0])
    print(f"Reference mean energy = {reference_energy:.6f} Ha")

    # ============================================================
    # 2. Run noisy experiments for each noise level
    # ============================================================
    energy_means, energy_stds, fidelity_means, fidelity_stds = [], [], [], []

    for p_dep, p_amp in zip(depolarizing_probs, amplitude_damping_probs):
        noisy_energies, fidelities = [], []
        for seed in seeds:
            np.random.seed(int(seed))
            res = run_vqe(
                molecule=molecule,
                n_steps=steps,
                stepsize=stepsize,
                plot=False,
                ansatz_name=ansatz_name,
                optimizer_name=optimizer_name,
                noisy=True,
                depolarizing_prob=p_dep,
                amplitude_damping_prob=p_amp,
                force=force,
                mapping=mapping,
            )
            noisy_energies.append(res["energy"])
            state = np.array(res["final_state_real"]) + 1j * np.array(res["final_state_imag"])
            state /= np.linalg.norm(state)
            fidelities.append(compute_fidelity(reference_state, state))

        noisy_energies = np.array(noisy_energies)
        ΔE = noisy_energies - reference_energy

        energy_means.append(np.mean(ΔE))
        energy_stds.append(np.std(ΔE))
        fidelity_means.append(np.mean(fidelities))
        fidelity_stds.append(np.std(fidelities))

        print(f"Noise p={p_dep:.2f}: ΔE = {np.mean(ΔE):.6f} ± {np.std(ΔE):.6f}, ⟨F⟩ = {np.mean(fidelities):.4f}")

    # ============================================================
    # 3. Plot summary statistics
    # ============================================================
    noise_levels = amplitude_damping_probs if noise_type == "amplitude" else depolarizing_probs
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
    )
    print(f"\n✅ Multi-seed noise study complete for {molecule}")


# ================================================================
# GEOMETRY SCANS
# ================================================================
def run_vqe_geometry_scan(
    molecule="H2_BOND",  # or "LiH_BOND", "H2O_ANGLE"
    param_name="bond",   # "bond" or "angle"
    param_values=None,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=30,
    stepsize=0.2,
    seeds=None,
    force=False,
    mapping: str = "jordan_wigner",
):
    """
    Scan a geometry parameter (bond length or angle) and plot mean ± std energy.

    Args:
        molecule: Parametric molecule identifier (e.g., "H2_BOND", "H2O_ANGLE").
        param_name: Name of the varying parameter.
        param_values: List/array of parameter values.
        ansatz_name, optimizer_name: Algorithmic choices.
        steps, stepsize, seeds: Optimization configuration.
    """
    import matplotlib.pyplot as plt

    if param_values is None:
        raise ValueError("param_values must be specified as a list or array")

    seeds = seeds or [0]
    results = []

    # ============================================================
    # 1. Loop over parameter values
    # ============================================================
    for val in param_values:
        print(f"\n⚙️ Geometry: {param_name} = {val:.3f}")
        symbols, coordinates = generate_geometry(molecule, val)

        energies_for_val = []
        for seed in seeds:
            np.random.seed(seed)
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
                seed=seed,
                force=force,
                mapping=mapping,
            )
            energies_for_val.append(res["energy"])

        mean_E = float(np.mean(energies_for_val))
        std_E = float(np.std(energies_for_val))
        results.append((val, mean_E, std_E))
        print(f"  → Mean E = {mean_E:.6f} ± {std_E:.6f} Ha")

    # ============================================================
    # 2. Plot energy vs geometry parameter
    # ============================================================
    params, means, stds = zip(*results)
    plt.errorbar(params, means, yerr=stds, fmt="o-", capsize=4)
    plt.xlabel(f"{param_name.capitalize()} (Å or °)")
    plt.ylabel("Ground-State Energy (Ha)")
    plt.title(f"{molecule} Energy vs {param_name.capitalize()} ({ansatz_name}, {optimizer_name})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(IMG_DIR, exist_ok=True)
    fname = f"{molecule}_Scan_{param_name}_{ansatz_name}_{optimizer_name}.png"
    plt.savefig(os.path.join(IMG_DIR, fname), dpi=300)
    plt.close()
    print(f"\n📉 Saved geometry-scan plot to {IMG_DIR}/{fname}")

    min_idx = int(np.argmin(means))
    print(f"Minimum energy: {means[min_idx]:.6f} ± {stds[min_idx]:.6f} at {param_name}={params[min_idx]:.3f}")

    return results


# ================================================================
# FERMION-TO-QUBIT MAPPING COMPARISONS
# ================================================================
def run_vqe_mapping_comparison(
    molecule="H2",
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    mappings=None,
    steps=50,
    stepsize=0.2,
    seed=0,
    force=False,
):
    """
    Compare different fermion-to-qubit mappings
    (Jordan–Wigner, Bravyi–Kitaev, Parity) for a fixed setup.
    """
    import matplotlib.pyplot as plt

    mappings = mappings or ["jordan_wigner", "bravyi_kitaev", "parity"]
    results = {}

    print(f"\n🔍 Comparing mappings for {molecule} ({ansatz_name}, {optimizer_name})")

    for mapping in mappings:
        print(f"\n⚙️ Running mapping: {mapping}")

        # --- Build Hamiltonian for this mapping ---
        result = build_hamiltonian(molecule, mapping=mapping)
        if len(result) == 5:
            H, qubits, symbols, coordinates, basis = result
        else:
            H, qubits = result
            symbols = coordinates = None
            basis = "sto-3g"

        # --- Config & caching ---
        cfg = make_run_config_dict(
            symbols=symbols if symbols is not None else [],
            coordinates=coordinates if coordinates is not None else [],
            basis=basis,
            ansatz_desc=ansatz_name,
            optimizer_name=optimizer_name,
            stepsize=stepsize,
            max_iterations=steps,
            seed=seed,
            mapping=mapping,
            noisy=False,
            depolarizing_prob=0.0,
            amplitude_damping_prob=0.0,
        )

        sig = run_signature(cfg)
        prefix = f"{molecule}_{mapping}_{optimizer_name}_s{seed}__{sig}"
        result_path = os.path.join(RESULTS_DIR, f"{prefix}.json")

        # --- Cached result check ---
        if not force and os.path.exists(result_path):
            print(f"📂 Using cached result for {mapping}")
            with open(result_path, "r") as f:
                record = json.load(f)
            cached = record.get("result", {})
            results[mapping] = {
                "final_energy": cached.get("energy"),
                "energies": cached.get("energies", []),
                "num_qubits": qubits,
                "num_terms": None,
            }
            continue

        # --- Run fresh optimization ---
        np.random.seed(seed)
        ansatz_fn = get_ansatz(ansatz_name)
        dev = qml.device("default.qubit", wires=qubits)

        @qml.qnode(dev)
        def circuit(params):
            ansatz_fn(params, wires=range(qubits))
            return qml.expval(H)

        params = init_params(ansatz_name, qubits, seed=seed)
        params, energies = minimize_energy(circuit, params, optimizer_name, steps=steps)
        final_E = float(energies[-1])

        # --- Extract Hamiltonian term count ---
        try:
            num_terms = len(H.ops)
        except AttributeError:
            try:
                num_terms = len(H.terms()[0]) if callable(H.terms) else len(H.data)
            except Exception:
                num_terms = len(H.data) if hasattr(H, "data") else None

        results[mapping] = {
            "final_energy": final_E,
            "energies": [float(e) for e in energies],
            "num_qubits": qubits,
            "num_terms": num_terms,
        }

        save_run_record(prefix, {"config": cfg, "result": {"energy": final_E, "energies": results[mapping]["energies"]}})

    # ============================================================
    # Plot mapping comparison
    # ============================================================
    os.makedirs(IMG_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for mapping, data in results.items():
        plt.plot(
            range(len(data["energies"])),
            data["energies"],
            label=mapping.replace("_", "-").title(),
            linewidth=2,
            alpha=0.9,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE: Energy Convergence by Mapping ({ansatz_name})")
    plt.legend(frameon=False, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=2)

    fname = f"{molecule}_Mapping_Convergence_{ansatz_name}_{optimizer_name}.png"
    plt.savefig(os.path.join(IMG_DIR, fname), dpi=300)
    plt.close()

    # --- Summary ---
    print(f"\n📉 Saved mapping comparison plot to {IMG_DIR}/{fname}\nResults Summary:")
    for mapping, data in results.items():
        print(f"  {mapping:15s} → E = {data['final_energy']:.8f} Ha, Qubits = {data['num_qubits']}, Terms = {data['num_terms']}")

    return results

