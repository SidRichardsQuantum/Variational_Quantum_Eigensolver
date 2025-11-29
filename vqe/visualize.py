"""
vqe.visualize
-------------
Unified plotting utilities for VQE and SSVQE.
All plots save into vqe/io_utils.IMG_DIR.

This module intentionally avoids external dependencies (e.g. common.plotting)
for maximum portability and internal cohesion.
"""

from __future__ import annotations

import os
import matplotlib.pyplot as plt

from .io_utils import IMG_DIR


# ================================================================
# INTERNAL HELPERS
# ================================================================
def _safe_filename(*parts):
    """
    Build a safe filename from components such as:
        ("VQE", "H2", "Adam", "UCCSD", "noisy")
    """
    clean = []
    for p in parts:
        if p is None:
            continue
        # Basic sanitisation
        p = str(p).replace(" ", "_").replace("+", "plus")
        clean.append(p)
    return "_".join(clean) + ".png"


def _safe_title(*parts):
    """
    Build a human-readable plot title.
    """
    return " — ".join([str(p) for p in parts if p is not None])


def _save_plot(fname):
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📁 Saved → {path}")


# ================================================================
# BASIC VQE CONVERGENCE
# ================================================================
def plot_convergence(
    energies_noiseless,
    molecule: str,
    energies_noisy=None,
    optimizer: str = "Adam",
    ansatz: str = "UCCSD",
    dep_prob: float = 0.0,
    amp_prob: float = 0.0,
):
    """
    Plot VQE energy convergence (noisy + noiseless overlay).
    """
    plt.figure(figsize=(8, 5))
    steps = range(len(energies_noiseless))
    plt.plot(steps, energies_noiseless, label="Noiseless", lw=2)

    noisy = energies_noisy is not None
    if noisy:
        plt.plot(range(len(energies_noisy)), energies_noisy,
                 label="Noisy", lw=2, linestyle="--")

    # Title
    if noisy:
        title = _safe_title(
            f"{molecule}",
            f"VQE Convergence ({optimizer}, {ansatz})",
            f"Noise: dep={dep_prob}, amp={amp_prob}",
        )
    else:
        title = _safe_title(
            f"{molecule}", f"VQE Convergence ({optimizer}, {ansatz})"
        )

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    # Filename
    fname = _safe_filename(
        "VQE_Convergence",
        molecule,
        optimizer,
        ansatz,
        "noisy" if noisy else "noiseless",
        f"dep{dep_prob}" if noisy else "",
        f"amp{amp_prob}" if noisy else "",
    )
    _save_plot(fname)


# ================================================================
# OPTIMIZER COMPARISON
# ================================================================
def plot_optimizer_comparison(molecule: str, results: dict, ansatz: str = "UCCSD"):
    """
    Plot multiple optimizers on a shared convergence graph.
    """
    plt.figure(figsize=(8, 5))

    min_len = min(len(v) for v in results.values())

    for opt, energies in results.items():
        plt.plot(range(min_len), energies[:min_len], label=opt)

    plt.title(_safe_title(molecule, f"VQE Optimizer Comparison ({ansatz})"))
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    fname = _safe_filename("VQE_Optimizer_Comparison", molecule, ansatz)
    _save_plot(fname)


# ================================================================
# ANSATZ COMPARISON
# ================================================================
def plot_ansatz_comparison(molecule: str, results: dict, optimizer: str = "Adam"):
    """
    Plot multiple ansatzes on a shared convergence graph.
    """
    plt.figure(figsize=(8, 5))

    min_len = min(len(v) for v in results.values())

    for ans, energies in results.items():
        plt.plot(range(min_len), energies[:min_len], label=ans)

    plt.title(_safe_title(molecule, f"VQE Ansatz Comparison ({optimizer})"))
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    fname = _safe_filename("VQE_Ansatz_Comparison", molecule, optimizer)
    _save_plot(fname)


# ================================================================
# NOISE STATISTICS
# ================================================================
def plot_noise_statistics(
    molecule: str,
    noise_levels,
    energy_means,
    energy_stds,
    fidelity_means,
    fidelity_stds,
    optimizer_name="Adam",
    ansatz_name="UCCSD",
    noise_type="Depolarizing",
):
    """
    Plot (ΔE vs noise) and (fidelity vs noise) as two subplots.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ΔE vs noise level
    ax1.errorbar(noise_levels, energy_means, yerr=energy_stds,
                 fmt="o-", capsize=4)
    ax1.set_ylabel("ΔE (Ha)")
    ax1.set_title(
        _safe_title(
            molecule,
            f"VQE Noise Impact — {noise_type}",
            f"{optimizer_name}, {ansatz_name}",
        )
    )
    ax1.grid(True, alpha=0.4)

    # Fidelity vs noise level
    ax2.errorbar(noise_levels, fidelity_means, yerr=fidelity_stds,
                 fmt="s-", capsize=4)
    ax2.set_xlabel("Noise Probability")
    ax2.set_ylabel("Fidelity")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()

    fname = _safe_filename(
        "VQE_Noise_Stats",
        molecule,
        optimizer_name,
        ansatz_name,
        noise_type,
    )
    _save_plot(fname)


# ---------------------------------------------------------------------
# SSVQE Multi-State Convergence Plot
# ---------------------------------------------------------------------
def plot_ssvqe_convergence_multi(
    energies_per_state,
    *,
    molecule="molecule",
    ansatz="UCCSD",
    optimizer="Adam",
    show=True,
    save=True,
):
    """
    Plot convergence for multiple states from SSVQE.

    Parameters
    ----------
    energies_per_state : dict or list
        Either:
            {0: [...], 1: [...], 2: [...]}  
        or a list-of-lists:
            [[...], [...], [...]]
        Each entry is the energy trajectory for one state.

    molecule : str
        Molecule label.
    ansatz : str
        Ansatz name.
    optimizer : str
        Optimizer name.
    show : bool
        Whether to display the plot.
    save : bool
        Whether to save the PNG via common.plotting.
    """

    import matplotlib.pyplot as plt
    from common.plotting import build_filename, save_plot, format_molecule_name

    # Normalise molecule name
    mol_norm = format_molecule_name(molecule)

    # Handle dict or list input
    if isinstance(energies_per_state, dict):
        trajectories = [energies_per_state[k] for k in sorted(energies_per_state.keys())]
    else:
        trajectories = energies_per_state

    n_states = len(trajectories)

    # Plot
    plt.figure(figsize=(7, 4.5))
    for i, E_list in enumerate(trajectories):
        plt.plot(E_list, label=f"State {i}")

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} SSVQE ({n_states} states) – {ansatz}, {optimizer}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save if requested
    if save:
        fname = build_filename(
            molecule=mol_norm,
            topic="ssvqe_convergence",
            extras={
                "states": n_states,
                "ans": ansatz,
                "opt": optimizer,
            }
        )
        save_plot(fname)

    if show:
        plt.show()
    else:
        plt.close()
