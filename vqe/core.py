import os
import json
import pennylane as qml
from pennylane import numpy as np
from .hamiltonian import build_hamiltonian
from .ansatz import get_ansatz, init_params
from .optimizer import minimize_energy
from .visualize import (
    plot_noise_sweep,
    plot_convergence,
    plot_optimizer_comparison,
    plot_ansatz_comparison,
)
from .io_utils import make_run_config_dict, run_signature, save_run_record


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
):
    """Run VQE workflow end-to-end with optional noise and caching."""

    np.random.seed(seed)

    # --- Build config signature and check cache ---
    H, qubits, symbols, coordinates, basis = build_hamiltonian(molecule)
    cfg = make_run_config_dict(
        symbols,
        coordinates,
        basis,
        ansatz_name,
        optimizer_name,
        0.4,
        n_steps,
        0,
        noisy,
        depolarizing_prob,
        amplitude_damping_prob,
    )
    sig = run_signature(cfg)
    prefix = f"{molecule}_{optimizer_name}_s0__{sig}"
    result_path = os.path.join("results", f"{prefix}.json")

    # --- Cache check ---
    if not force and os.path.exists(result_path):
        print(f"\n📂 Found cached result for this configuration: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # --- Prepare circuit ---
    ansatz_fn = get_ansatz(ansatz_name)
    dev_name = "default.mixed" if noisy else "default.qubit"
    diff_method = "finite-diff" if noisy else "parameter-shift"
    dev = qml.device(dev_name, wires=qubits)

    @qml.qnode(dev, diff_method=diff_method)
    def circuit(params):
        ansatz_fn(params, wires=range(qubits))
        if noisy:
            for w in range(qubits):
                if depolarizing_prob > 0:
                    qml.DepolarizingChannel(depolarizing_prob, wires=w)
                if amplitude_damping_prob > 0:
                    qml.AmplitudeDamping(amplitude_damping_prob, wires=w)
        return qml.expval(H)

    @qml.qnode(dev, diff_method=diff_method)
    def get_state(params):
        ansatz_fn(params, wires=range(qubits))
        if noisy:
            for w in range(qubits):
                if depolarizing_prob > 0:
                    qml.DepolarizingChannel(depolarizing_prob, wires=w)
                if amplitude_damping_prob > 0:
                    qml.AmplitudeDamping(amplitude_damping_prob, wires=w)
        return qml.state()

    # --- Initialize parameters ---
    np.random.seed(0)
    params = init_params(ansatz_name, qubits)

    # --- Optimization ---
    params, energies = minimize_energy(
        circuit,
        params,
        optimizer_name,
        steps=n_steps,
    )

    final_energy = float(energies[-1])
    final_state = get_state(params)

    # --- Plot ---
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
            force=args.force,
        )
        results.append((p_dep, res["energies"]))

    plot_noise_sweep(
        molecule,
        results,
        optimizer=optimizer_name,
        ansatz=ansatz_name,
    )
    print(f"\n✅ Completed noise sweep for {molecule}")


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
    """Compare multiple optimizers for the same molecule and ansatz."""
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
    """Compare multiple ansatz circuits for the same molecule and optimizer."""
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


def compute_fidelity(pure_state, state_or_rho):
    """Compute fidelity between a pure state and a (statevector or density matrix)."""
    import numpy as np
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
    """Run VQE for multiple seeds & noise levels and compute mean ± std of energy errors and fidelities."""
    from .visualize import plot_noise_statistics
    from pennylane import numpy as np

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
            depolarizing_prob=0.0,
            amplitude_damping_prob=0.0,
            force=force,
        )
        ref_energies.append(res["energy"])
        state = np.array(res["final_state_real"]) + 1j * np.array(res["final_state_imag"])
        ref_states.append(state)

    reference_energy = float(np.mean(ref_energies))
    reference_state = ref_states[0] / np.linalg.norm(ref_states[0])
    print(f"Reference (mean noiseless) energy = {reference_energy:.6f} Ha")

    # --- noisy runs & stats ---
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

    # --- summarize & plot ---
    if noise_type == "amplitude":
        noise_levels = amplitude_damping_probs
    else:
        noise_levels = depolarizing_probs

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
