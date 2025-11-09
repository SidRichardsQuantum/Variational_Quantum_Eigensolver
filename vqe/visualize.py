"""
vqe.visualize
-------------
Plotting utilities for Variational Quantum Eigensolver (VQE) results.

Includes:
- Energy convergence curves
- Noise-level comparisons
- Optimizer and ansatz comparisons
- Fidelity/error vs noise statistics
- SSVQE excited-state plots
"""

import os
import matplotlib.pyplot as plt
from .io_utils import IMG_DIR


# ================================================================
# HELPER
# ================================================================
def _save_plot(fname: str):
    """Save current matplotlib figure to IMG_DIR and close."""
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n📉 Saved plot to: {path}")


# ================================================================
# CORE VQE PLOTS
# ================================================================
def plot_convergence(
    energies_noiseless,
    molecule: str,
    energies_noisy=None,
    optimizer: str = "Adam",
    ansatz: str = "RY-CZ",
    dep_prob: float = 0.0,
    amp_prob: float = 0.0,
    noisy: bool = False,
):
    """
    Plot energy convergence curve for a VQE run (noisy and/or noiseless).

    Args:
        energies_noiseless: Energy values for noiseless run.
        molecule: Molecule label (used in title and filename).
        energies_noisy: Optional list of noisy run energies.
        optimizer: Optimizer name.
        ansatz: Ansatz name.
        dep_prob: Depolarizing noise probability.
        amp_prob: Amplitude damping probability.
        noisy: Whether the run included noise.
    """
    plt.figure(figsize=(8, 6))
    steps = range(1, len(energies_noiseless) + 1)
    plt.plot(steps, energies_noiseless, "b-", lw=2, label="Noiseless")

    if energies_noisy is not None:
        plt.plot(range(1, len(energies_noisy) + 1), energies_noisy, "r--", lw=2, label="Noisy")

    if noisy:
        title = (
            f"{molecule} VQE Convergence: "
            f"Depolarizing (p={dep_prob}) + Amplitude Damping (p={amp_prob})\n"
            f"({optimizer}, {ansatz})"
        )
        fname = f"{molecule}_Noise_Comparison_{optimizer}_{ansatz}.png"
    else:
        title = f"{molecule} VQE Convergence ({optimizer}, {ansatz})"
        fname = f"{molecule}_VQE_Convergence.png"

    plt.title(title, fontsize=12)
    plt.xlabel("Iteration", fontsize=11)
    plt.ylabel("Energy (Ha)", fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    _save_plot(fname)


# ================================================================
# NOISE STUDY PLOTS
# ================================================================
def plot_noise_sweep(molecule: str, results, optimizer="Adam", ansatz="RY-CZ"):
    """Plot energy convergence vs iteration for multiple noise levels."""
    plt.figure(figsize=(10, 6))
    min_len = min(len(e[1]) for e in results)

    for p, energies in results:
        plt.plot(range(1, min_len + 1), energies[:min_len], label=f"p={p:.2f}")

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE Convergence vs Noise Level\n({optimizer}, {ansatz})")
    plt.legend(title="Depolarizing prob.")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    _save_plot(f"{molecule}_Noise_Sweep_{optimizer}_{ansatz}.png")


def plot_noise_statistics(
    molecule: str,
    noise_levels,
    energy_means,
    energy_stds,
    fidelity_means,
    fidelity_stds,
    optimizer_name="Adam",
    ansatz_name="RY-CZ",
    noise_type="Depolarizing",
):
    """Plot mean ± std of energy error and fidelity vs noise probability."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # --- Energy error subplot ---
    ax1.errorbar(
        noise_levels, energy_means, yerr=energy_stds,
        fmt="o-", capsize=4, label="Energy Error"
    )
    ax1.set_ylabel("ΔE (Ha)")
    ax1.set_title(f"{molecule} Energy Error vs Noise ({optimizer_name}, {ansatz_name})")
    ax1.grid(True, alpha=0.4)
    ax1.legend()

    # --- Fidelity subplot ---
    ax2.errorbar(
        noise_levels, fidelity_means, yerr=fidelity_stds,
        fmt="s-", capsize=4, color="tab:orange", label="Fidelity"
    )
    ax2.set_xlabel("Noise Probability")
    ax2.set_ylabel("Fidelity |⟨ψ₀|ψ⟩|²")
    ax2.set_title(f"{molecule} Fidelity vs Noise ({optimizer_name}, {ansatz_name})")
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    _save_plot(f"{molecule}_{noise_type.capitalize()}_Noise_Stats_{optimizer_name}_{ansatz_name}.png")


# ================================================================
# COMPARISON PLOTS
# ================================================================
def plot_optimizer_comparison(molecule: str, results: dict, ansatz="RY-CZ"):
    """Compare optimizer performance on VQE convergence."""
    plt.figure(figsize=(10, 6))
    min_len = min(len(v) for v in results.values())

    for opt, energies in results.items():
        plt.plot(range(1, min_len + 1), energies[:min_len], label=opt)

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE Optimizer Comparison ({ansatz})")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    _save_plot(f"{molecule}_Optimizer_Comparison_{ansatz}.png")


def plot_ansatz_comparison(molecule: str, results: dict, optimizer="Adam"):
    """Compare multiple ansatz circuits under a fixed optimizer."""
    plt.figure(figsize=(10, 6))
    min_len = min(len(v) for v in results.values())

    for ans, energies in results.items():
        plt.plot(range(1, min_len + 1), energies[:min_len], label=ans)

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE Ansatz Comparison ({optimizer})")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    _save_plot(f"{molecule}_Ansatz_Comparison_{optimizer}.png")


# ================================================================
# SSVQE PLOTS
# ================================================================
def plot_ssvqe_convergence(molecule: str, E0_list, E1_list, optimizer_name="Adam"):
    """Plot ground and first excited state energies over iterations."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(E0_list) + 1), E0_list, label="Ground (E₀)")
    plt.plot(range(1, len(E1_list) + 1), E1_list, label="1st Excited (E₁)")

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} SSVQE Convergence ({optimizer_name})")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    fname = f"{molecule.replace('+','plus')}_SSVQE_{optimizer_name}.png"
    _save_plot(fname)


def plot_ssvqe_convergence_multi(
    molecule: str,
    energies_per_state,
    labels=None,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
):
    """Plot convergence for multiple eigenstates (multi-state SSVQE)."""
    if labels is None:
        labels = [f"E{i}" for i in range(len(energies_per_state))]

    plt.figure(figsize=(8, 6))
    for k, E in enumerate(energies_per_state):
        plt.plot(range(1, len(E) + 1), E, label=labels[k])

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} SSVQE Convergence ({ansatz_name}, {optimizer_name})")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    fname = f"{molecule.replace('+','plus')}_SSVQE_{ansatz_name}_{optimizer_name}.png"
    _save_plot(fname)
