import os
import json
import pennylane as qml
from pennylane import numpy as np
from .hamiltonian import build_hamiltonian, generate_geometry
from .ansatz import get_ansatz, init_params
from .optimizer import minimize_energy
from .visualize import (
    plot_noise_sweep,
    plot_convergence,
    plot_optimizer_comparison,
    plot_ansatz_comparison,
)
from .io_utils import IMG_DIR, make_run_config_dict, run_signature, save_run_record, ensure_dirs
ensure_dirs()

# ================================================================
# MAIN VQE EXECUTION
# ================================================================
def run_vqe(
    molecule: str = "H2",
    seed: int = 0,
    n_steps: int = 50,
    plot: bool = True,
    ansatz_name: str = "StronglyEntanglingLayers",
    optimizer_name: str = "GradientDescent",
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
):
    """Run VQE workflow end-to-end with optional noise and caching."""

    np.random.seed(seed)

    # --- Build or reuse molecular Hamiltonian ---
    if symbols is not None and coordinates is not None:
        H, qubits = qml.qchem.molecular_hamiltonian(
            symbols,
            coordinates,
            charge=0,
            basis=basis,
            unit="angstrom"
        )
    else:
        H, qubits, symbols, coordinates, basis = build_hamiltonian(molecule)

    ansatz_fn = get_ansatz(ansatz_name)

    # --- Build config & check cache ---
    cfg = make_run_config_dict(
        symbols,
        coordinates,
        basis,
        ansatz_name,
        optimizer_name,
        0.4,
        n_steps,
        seed,
        noisy,
        depolarizing_prob,
        amplitude_damping_prob,
    )
    sig = run_signature(cfg)
    prefix = f"{molecule}_{optimizer_name}_s{seed}__{sig}"
    result_path = os.path.join("results", f"{prefix}.json")

    if not force and os.path.exists(result_path):
        print(f"\n📂 Found cached result for this configuration: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # --- Device and QNode setup ---
    dev_name = "default.mixed" if noisy else "default.qubit"
    diff_method = "finite-diff" if noisy else "parameter-shift"
    dev = qml.device(dev_name, wires=qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params):
        ansatz_fn(params, wires=range(qubits), symbols=symbols, coordinates=coordinates, basis=basis)
        if noisy:
            for w in range(qubits):
                if depolarizing_prob > 0:
                    qml.DepolarizingChannel(depolarizing_prob, wires=w)
                if amplitude_damping_prob > 0:
                    qml.AmplitudeDamping(amplitude_damping_prob, wires=w)
        return qml.expval(H)

    @qml.qnode(dev, diff_method=diff_method)
    def get_state(params):
        ansatz_fn(params, wires=range(qubits), symbols=symbols, coordinates=coordinates, basis=basis)
        if noisy:
            for w in range(qubits):
                if depolarizing_prob > 0:
                    qml.DepolarizingChannel(depolarizing_prob, wires=w)
                if amplitude_damping_prob > 0:
                    qml.AmplitudeDamping(amplitude_damping_prob, wires=w)
        return qml.state()

    # --- Initialize parameters properly ---
    params = init_params(
        ansatz_name,
        num_wires=qubits,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )

    # --- Optimization ---
    params, energies = minimize_energy(circuit, params, optimizer_name, steps=n_steps)
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
):
    """Run VQE for a range of noise levels and plot all curves on one figure."""
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
    steps=30,
    noisy=True,
    depolarizing_prob=0.02,
    amplitude_damping_prob=0.01,
    force=False,
):
    optimizers = optimizers or ["Adam", "GradientDescent", "Momentum"]
    results = {}
    for opt_name in optimizers:
        print(f"\n⚙️  Running optimizer: {opt_name}")
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            ansatz_name=ansatz_name,
            optimizer_name=opt_name,
            noisy=noisy,
            depolarizing_prob=depolarizing_prob,
            amplitude_damping_prob=amplitude_damping_prob,
            plot=False,
            force=force,
        )
        results[opt_name] = res["energies"]

    plot_optimizer_comparison(molecule, results, ansatz=ansatz_name)
    print(f"\n✅ Optimizer comparison complete for {molecule} ({ansatz_name})")


def run_vqe_ansatz_comparison(
    molecule="H2",
    optimizer_name="Adam",
    ansatzes=None,
    steps=30,
    noisy=True,
    depolarizing_prob=0.02,
    amplitude_damping_prob=0.01,
    force=False,
):
    ansatzes = ansatzes or ["RY-CZ", "Minimal", "TwoQubit-RY-CNOT"]
    results = {}
    for ans_name in ansatzes:
        print(f"\n🔹 Running ansatz: {ans_name}")
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            ansatz_name=ans_name,
            optimizer_name=optimizer_name,
            noisy=noisy,
            depolarizing_prob=depolarizing_prob,
            amplitude_damping_prob=amplitude_damping_prob,
            plot=False,
            force=force,
        )
        results[ans_name] = res["energies"]

    plot_ansatz_comparison(molecule, results, optimizer=optimizer_name)
    print(f"\n✅ Ansatz comparison complete for {molecule} ({optimizer_name})")


# ================================================================
# FIDELITY & MULTI-SEED NOISE
# ================================================================
def compute_fidelity(pure_state, state_or_rho):
    state_or_rho = np.array(state_or_rho)
    pure_state = np.array(pure_state)
    if state_or_rho.ndim == 1:
        return float(abs(np.vdot(pure_state, state_or_rho)) ** 2)
    elif state_or_rho.ndim == 2:
        return float(np.real(np.vdot(pure_state, state_or_rho @ pure_state)))
    else:
        raise ValueError("Invalid state shape for fidelity computation")


def run_vqe_multi_seed_noise(
    molecule="H2",
    ansatz_name="RY-CZ",
    optimizer_name="Adam",
    steps=30,
    seeds=None,
    noise_type="depolarizing",
    depolarizing_probs=None,
    amplitude_damping_probs=None,
    force=False,
):
    from .visualize import plot_noise_statistics

    seeds = seeds or np.arange(0, 5)
    if depolarizing_probs is None:
        depolarizing_probs = np.arange(0.0, 0.11, 0.02)

    # Configure which noise to apply
    if noise_type == "depolarizing":
        amplitude_damping_probs = [0.0] * len(depolarizing_probs)
    elif noise_type == "amplitude":
        amplitude_damping_probs = depolarizing_probs
        depolarizing_probs = [0.0] * len(amplitude_damping_probs)
    elif noise_type == "combined":
        amplitude_damping_probs = depolarizing_probs.copy()
    else:
        raise ValueError(f"Unknown noise type '{noise_type}'")

    # --- noiseless reference runs ---
    print("\n🔹 Computing noiseless reference runs...")
    ref_energies, ref_states = [], []
    for seed in seeds:
        np.random.seed(int(seed))
        res = run_vqe(
            molecule=molecule,
            n_steps=steps,
            plot=False,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            noisy=False,
            force=force,
        )
        ref_energies.append(res["energy"])
        state = np.array(res["final_state_real"]) + 1j * np.array(res["final_state_imag"])
        ref_states.append(state)

    reference_energy = float(np.mean(ref_energies))
    reference_state = ref_states[0] / np.linalg.norm(ref_states[0])
    print(f"Reference (mean noiseless) energy = {reference_energy:.6f} Ha")

    # --- noisy runs ---
    energy_means, energy_stds, fidelity_means, fidelity_stds = [], [], [], []
    for p_dep, p_amp in zip(depolarizing_probs, amplitude_damping_probs):
        noisy_energies, fidelities = [], []
        for seed in seeds:
            np.random.seed(int(seed))
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
            )
            noisy_energies.append(res["energy"])
            state = np.array(res["final_state_real"]) + 1j * np.array(res["final_state_imag"])
            state /= np.linalg.norm(state)
            fidelity = compute_fidelity(reference_state, state)
            fidelities.append(fidelity)

        noisy_energies = np.array(noisy_energies)
        ΔE = noisy_energies - reference_energy
        energy_means.append(np.mean(ΔE))
        energy_stds.append(np.std(ΔE))
        fidelity_means.append(np.mean(fidelities))
        fidelity_stds.append(np.std(fidelities))

        print(f"Noise p={p_dep:.2f}: ΔE = {np.mean(ΔE):.6f} ± {np.std(ΔE):.6f}, ⟨F⟩ = {np.mean(fidelities):.4f}")

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
    molecule="H2O_ANGLE",
    param_name="angle",
    param_values=None,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=30,
    seeds=None,
    force=False,
):
    """Run VQE across a range of molecular geometries (e.g. bond angles or lengths),
    reusing cached results when available.
    """

    if param_values is None:
        raise ValueError("param_values must be specified as a list or array")

    seeds = seeds or [0]
    results = []

    for param_value in param_values:
        print(f"\n⚙️ Running geometry: {param_name} = {param_value:.2f}")
        symbols, coordinates = generate_geometry(molecule, param_value)
        energies_for_param = []

        for seed in seeds:
            np.random.seed(seed)

            # ---- check for existing cached run ----
            cfg = make_run_config_dict(
                symbols,
                coordinates,
                "sto-3g",
                ansatz_name,
                optimizer_name,
                0.4,
                steps,
                seed,
                noisy=False,
                depolarizing_prob=0.0,
                amplitude_damping_prob=0.0,
            )
            sig = run_signature(cfg)
            from .io_utils import build_run_filename
            fname = build_run_filename(
                f"{molecule}_{param_name}_{param_value:.3f}",
                optimizer_name,
                seed,
                sig,
            )
            existing = find_existing_run(sig)

            if existing and not force:
                print(f"📂 Using cached result: {existing}")
                with open(existing) as f:
                    rec = json.load(f)
                E = float(rec["result"]["energy"])
                energies_for_param.append(E)
                continue

            # ---- perform VQE if not cached ----
            res = run_vqe(
                molecule=molecule,
                n_steps=steps,
                ansatz_name=ansatz_name,
                optimizer_name=optimizer_name,
                symbols=symbols,
                coordinates=coordinates,
                noisy=False,
                plot=False,
                seed=seed,
                force=force,
            )

            # save lightweight record (mirrors notebook behaviour)
            record = {
                "config": cfg,
                "result": res,
                "metadata": {param_name: float(param_value)},
            }
            persisted = save_run_record(fname, record)
            print(f"💾 Saved geometry run to {persisted}")

            energies_for_param.append(res["energy"])

        # ---- aggregate mean & std across seeds ----
        mean_E = np.mean(energies_for_param)
        std_E = np.std(energies_for_param)
        results.append((param_value, mean_E, std_E))
        print(f"  → Mean E = {mean_E:.6f} ± {std_E:.6f} Ha")

    # ---- plot energy profile ----
    params, means, stds = zip(*results)
    plt.errorbar(params, means, yerr=stds, fmt="o-", capsize=4)
    plt.xlabel(f"{param_name.capitalize()} (° or Å)")
    plt.ylabel("Ground State Energy (Ha)")
    plt.title(f"{molecule} Energy vs. {param_name.capitalize()} ({ansatz_name}, {optimizer_name})")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    fname = f"{molecule}_Scan_{param_name}_{ansatz_name}_{optimizer_name}.png"
    plt.savefig(f"{IMG_DIR}/{fname}", dpi=300)
    plt.close()
    print(f"\n📉 Saved geometry-scan plot to {IMG_DIR}/{fname}")

    # ---- report optimal geometry ----
    min_idx = np.argmin(means)
    print(f"Minimum energy: {means[min_idx]:.6f} ± {stds[min_idx]:.6f} at {param_name}={params[min_idx]:.2f}")

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
    seed=0,
    force=False,
):
    """Compare different fermion-to-qubit mappings (Jordan–Wigner, Bravyi–Kitaev, Parity)
    for the same molecule, ansatz, and optimizer.
    """
    import matplotlib.pyplot as plt
    import pennylane as qml
    from .hamiltonian import build_hamiltonian
    from .ansatz import get_ansatz, init_params
    from .optimizer import minimize_energy
    from .io_utils import IMG_DIR, make_run_config_dict, run_signature, save_run_record
    import os
    import json

    mappings = mappings or ["jordan_wigner", "bravyi_kitaev", "parity"]
    results = {}

    print(f"\n🔍 Comparing mappings for {molecule} ({ansatz_name}, {optimizer_name})")

    for mapping in mappings:
        print(f"\n⚙️  Running mapping: {mapping}")

        # --- Build Hamiltonian for this mapping ---
        result = build_hamiltonian(molecule, mapping=mapping)
        if len(result) == 5:
            H, qubits, symbols, coordinates, basis = result
        else:
            H, qubits = result
            symbols = coordinates = None
            basis = "sto-3g"

        # --- Build config and cache key ---
        cfg = make_run_config_dict(
            symbols if symbols is not None else [],
            coordinates if coordinates is not None else [],
            basis,
            ansatz_name,
            optimizer_name,
            0.4,
            steps,
            seed,
            noisy=False,
            depolarizing_prob=0.0,
            amplitude_damping_prob=0.0,
        )
        sig = run_signature(cfg)
        prefix = f"{molecule}_{mapping}_{optimizer_name}_s{seed}__{sig}"
        result_path = os.path.join("results", f"{prefix}.json")

        # --- Load cached result if available ---
        if not force and os.path.exists(result_path):
            print(f"📂 Using cached result for {mapping}: {result_path}")
            with open(result_path, "r") as f:
                record = json.load(f)
            cached = record.get("result", {})

            # Normalize naming to match new format
            results[mapping] = {
                "final_energy": cached.get("energy", None),
                "energies": cached.get("energies", []),
                "num_qubits": cached.get("num_qubits", None),
                "num_terms": cached.get("num_terms", None),
            }
            continue

        # --- Prepare ansatz and circuit ---
        np.random.seed(seed)
        ansatz_fn = get_ansatz(ansatz_name)
        dev = qml.device("default.qubit", wires=qubits)

        @qml.qnode(dev)
        def circuit(params):
            ansatz_fn(params, wires=range(qubits))
            return qml.expval(H)

        # --- Optimize ---
        params = init_params(ansatz_name, qubits, seed=seed)
        params, energies = minimize_energy(
            circuit,
            params,
            optimizer_name,
            steps=steps,
        )

        # --- Collect statistics ---
        try:
            num_terms = len(H.ops)
        except AttributeError:
            try:
                num_terms = len(H.terms()[0]) if callable(H.terms) else len(H.data)
            except Exception:
                num_terms = len(H.data) if hasattr(H, "data") else None

        results[mapping] = {
            "final_energy": float(energies[-1]),
            "energies": [float(e) for e in energies],
            "num_qubits": qubits,
            "num_terms": num_terms,
        }

        # --- Save record for caching ---
        record = {
            "config": cfg,
            "result": {
                "energy": float(energies[-1]),
                "energies": [float(e) for e in energies],
            },
        }
        save_run_record(prefix, record)

    # --- Plot comparison ---
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
    plt.title(f"{molecule} VQE: Energy Convergence by Mapping (Noiseless, {ansatz_name})")
    plt.legend(frameon=False, fontsize=10)
    plt.minorticks_on()
    plt.grid(True, alpha=0.3)

    plot_fname = f"{IMG_DIR}/{molecule}_Mapping_Convergence_{ansatz_name}_{optimizer_name}.png"
    plt.tight_layout(pad=2)
    plt.savefig(plot_fname, dpi=300)
    plt.close()

    print(f"\n📉 Saved mapping comparison convergence plot to {plot_fname}")

    # --- Print results summary ---
    print("\nResults Summary:")
    for mapping, data in results.items():
        print(
            f"  {mapping:15s} → E = {data['final_energy']:.8f} Ha, "
            f"Qubits = {data['num_qubits']}, Terms = {data['num_terms']}"
        )

    return results
