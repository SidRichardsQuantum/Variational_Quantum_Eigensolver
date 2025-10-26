import matplotlib.pyplot as plt
import os
from .io_utils import IMG_DIR


def plot_convergence(
    energies_noiseless,
    molecule,
    energies_noisy=None,
    optimizer="Adam",
    ansatz="RY-CZ",
    dep_prob=0.0,
    amp_prob=0.0,
    noisy=False,
):
    """Plot VQE energy convergence curves (noisy and/or noiseless) in consistent style."""
    plt.figure(figsize=(8, 6))
    steps = range(1, len(energies_noiseless) + 1)

    # Base noiseless curve
    plt.plot(steps, energies_noiseless, "b-", lw=2, label="Noiseless")

    # Optional noisy overlay
    if energies_noisy is not None:
        plt.plot(range(1, len(energies_noisy) + 1), energies_noisy, "r--", lw=2, label="Noisy")

    # Titles and labels
    if noisy:
        title = (
            f"{molecule} VQE Convergence: Depolarizing (p={dep_prob}) + "
            f"Amplitude Damping (p={amp_prob}) Noise\n({optimizer}, {ansatz})"
        )
        fname = f"{molecule}_Noise_Comparison_{optimizer}_{ansatz}.png"
    else:
        title = f"{molecule} VQE Convergence ({optimizer}, {ansatz})"
        fname = f"{molecule}_VQE_Convergence.png"

    plt.title(title, fontsize=12)
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Energy (Ha)", fontsize=11)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n📉 Saved convergence plot to: {path}")


def plot_noise_sweep(molecule, results, optimizer="Adam", ansatz="RY-CZ"):
    """Plot VQE energy convergence vs iteration for multiple noise levels."""
    plt.figure(figsize=(10, 6))
    min_len = min(len(e[1]) for e in results)
    for p, energies in results:
        plt.plot(range(1, min_len + 1), energies[:min_len], label=f"p={p:.2f}")

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE Convergence vs Noise Level\n({optimizer}, {ansatz})")
    plt.legend(title="Depolarizing prob.")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(IMG_DIR, exist_ok=True)
    fname = f"{molecule}_Noise_Sweep_{optimizer}_{ansatz}.png"
    path = os.path.join(IMG_DIR, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n📉 Saved noise sweep plot to: {path}")


def plot_optimizer_comparison(molecule, results, ansatz="RY-CZ"):
    """Plot fidelity or convergence for multiple optimizers."""
    plt.figure(figsize=(10, 6))
    min_len = min(len(v) for v in results.values())

    for opt, energies in results.items():
        plt.plot(range(1, min_len + 1), energies[:min_len], label=opt)

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE Convergence: Optimizer Comparison ({ansatz})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(IMG_DIR, exist_ok=True)
    fname = f"{molecule}_Optimizer_Comparison_{ansatz}.png"
    path = os.path.join(IMG_DIR, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n📉 Saved optimizer comparison plot to: {path}")


def plot_ansatz_comparison(molecule, results, optimizer="Adam"):
    """Plot energy convergence for multiple ansatz circuits."""
    plt.figure(figsize=(10, 6))
    min_len = min(len(v) for v in results.values())

    for ans, energies in results.items():
        plt.plot(range(1, min_len + 1), energies[:min_len], label=ans)

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE Convergence: Ansatz Comparison ({optimizer})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(IMG_DIR, exist_ok=True)
    fname = f"{molecule}_Ansatz_Comparison_{optimizer}.png"
    path = os.path.join(IMG_DIR, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n📉 Saved ansatz comparison plot to: {path}")


def plot_noise_statistics(
    molecule,
    noise_levels,
    energy_means,
    energy_stds,
    fidelity_means,
    fidelity_stds,
    optimizer_name="Adam",
    ansatz_name="RY-CZ",
    noise_type="Depolarizing",
):
    """Plot mean ± std of energy error and fidelity vs noise strength."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Energy error subplot
    ax1.errorbar(noise_levels, energy_means, yerr=energy_stds, fmt="o-", capsize=4, label="Energy Error")
    ax1.set_ylabel("ΔE (Ha)")
    ax1.set_title(f"{molecule} Energy Error vs Noise Strength ({optimizer_name}, {ansatz_name})")
    ax1.grid(True)
    ax1.legend()

    # Fidelity subplot
    ax2.errorbar(noise_levels, fidelity_means, yerr=fidelity_stds, fmt="s-", capsize=4, color="tab:orange", label="Fidelity")
    ax2.set_xlabel("Noise Probability")
    ax2.set_ylabel("Fidelity |⟨ψ₀|ψ⟩|²")
    ax2.set_title(f"{molecule} Fidelity vs Noise Strength ({optimizer_name}, {ansatz_name})")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    fname = f"{molecule}_{noise_type.capitalize()}_Noise_Error_{optimizer_name}_{ansatz_name}.png"
    path = os.path.join(IMG_DIR, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n📉 Saved noise-statistics plot to: {path}")


def plot_ssvqe_convergence(molecule, E0_list, E1_list, optimizer_name="Adam"):
    import matplotlib.pyplot as plt, os
    from .io_utils import IMG_DIR
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(E0_list)+1), E0_list, label="Ground (E₀)")
    plt.plot(range(1, len(E1_list)+1), E1_list, label="1st Excited (E₁)")
    plt.xlabel("Iteration"); plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} SSVQE Convergence ({optimizer_name})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, f"{molecule.replace('+','plus')}_SSVQE_{optimizer_name}.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"\n📉 Saved SSVQE convergence plot to: {path}")


def plot_ssvqe_convergence_multi(molecule, energies_per_state, labels=None, ansatz_name="UCCSD", optimizer_name="Adam"):
    import matplotlib.pyplot as plt, os
    from .io_utils import IMG_DIR
    if labels is None:
        labels = [f"E{i}" for i in range(len(energies_per_state))]
    plt.figure(figsize=(8, 6))
    for k, E in enumerate(energies_per_state):
        plt.plot(range(1, len(E)+1), E, label=labels[k])
    plt.xlabel("Iteration"); plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} SSVQE Convergence ({ansatz_name}, {optimizer_name})")
    plt.legend(); plt.grid(True); plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, f"{molecule.replace('+','plus')}_SSVQE_{ansatz_name}_{optimizer_name}.png")
    plt.savefig(path, dpi=300); plt.close()
    print(f"\n📉 Saved SSVQE convergence (multi) plot to: {path}")
