"""
vqe.visualize
-------------
Unified plotting utilities for VQE using common.plotting.
"""

import os
import matplotlib.pyplot as plt
from common.plotting import (
    build_filename,
    build_title,
    save_plot,
    normalize_molecule_name,
    pretty_molecule_name,
)
from .io_utils import IMG_DIR

# ---------------------------------------------------------------
# INTERNAL SAVE WRAPPER
# ---------------------------------------------------------------

def _save(fname: str):
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {path}")

# ---------------------------------------------------------------
# VQE CONVERGENCE
# ---------------------------------------------------------------

def plot_convergence(
    energies_noiseless,
    molecule: str,
    energies_noisy=None,
    optimizer: str = "Adam",
    ansatz: str = "RY-CZ",
    dep_prob: float = 0.0,
    amp_prob: float = 0.0,
):
    mol_norm = normalize_molecule_name(molecule)
    mol_pretty = pretty_molecule_name(molecule)

    plt.figure(figsize=(8, 6))
    steps = range(len(energies_noiseless))
    plt.plot(steps, energies_noiseless, label="Noiseless", lw=2)

    noisy = energies_noisy is not None
    if noisy:
        plt.plot(range(len(energies_noisy)), energies_noisy, label="Noisy", lw=2, linestyle="--")

    # Title
    if noisy:
        title = build_title(
            molecule,
            f"VQE: Convergence ({optimizer}, {ansatz})",
            f"Noise p_dep={dep_prob}, p_amp={amp_prob}",
        )
    else:
        title = build_title(
            molecule,
            f"VQE: Convergence ({optimizer}, {ansatz})",
        )

    # Filename
    if noisy:
        fname = build_filename(
            "VQE_Convergence",
            mol_norm,
            optimizer,
            ansatz,
            f"dep{dep_prob}",
            f"amp{amp_prob}",
        )
    else:
        fname = build_filename(
            "VQE_Convergence",
            mol_norm,
            optimizer,
            ansatz,
            "noiseless",
        )

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    _save(fname)

# ---------------------------------------------------------------
# OPTIMIZER COMPARISON
# ---------------------------------------------------------------

def plot_optimizer_comparison(molecule: str, results: dict, ansatz="RY-CZ"):
    mol_norm = normalize_molecule_name(molecule)

    plt.figure(figsize=(8, 6))
    min_len = min(len(v) for v in results.values())

    for opt, energies in results.items():
        plt.plot(range(min_len), energies[:min_len], label=opt)

    title = build_title(molecule, f"VQE: Optimizer Comparison ({ansatz})")
    fname = build_filename("VQE_Optimizer_Comparison", mol_norm, ansatz)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fname)

# ---------------------------------------------------------------
# ANSATZ COMPARISON
# ---------------------------------------------------------------

def plot_ansatz_comparison(molecule: str, results: dict, optimizer="Adam"):
    mol_norm = normalize_molecule_name(molecule)

    plt.figure(figsize=(8, 6))
    min_len = min(len(v) for v in results.values())

    for ans, energies in results.items():
        plt.plot(range(min_len), energies[:min_len], label=ans)

    title = build_title(molecule, f"VQE: Ansatz Comparison ({optimizer})")
    fname = build_filename("VQE_Ansatz_Comparison", mol_norm, optimizer)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fname)

# ---------------------------------------------------------------
# NOISE STATISTICS
# ---------------------------------------------------------------

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
    mol_norm = normalize_molecule_name(molecule)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.errorbar(noise_levels, energy_means, yerr=energy_stds, fmt="o-", capsize=4)
    ax1.set_ylabel("ΔE (Ha)")
    ax1.set_title(build_title(molecule, f"VQE: Energy Error vs {noise_type}"))
    ax1.grid(True, alpha=0.4)

    ax2.errorbar(noise_levels, fidelity_means, yerr=fidelity_stds, fmt="s-", capsize=4)
    ax2.set_xlabel("Noise Probability")
    ax2.set_ylabel("Fidelity |⟨ψ₀|ψ⟩|²")
    ax2.set_title(build_title(molecule, f"VQE: Fidelity vs {noise_type}"))
    ax2.grid(True, alpha=0.4)

    fname = build_filename(
        "VQE_Noise_Stats",
        mol_norm,
        optimizer_name,
        ansatz_name,
        noise_type,
    )
    plt.tight_layout()
    _save(fname)

# ---------------------------------------------------------------
# SSVQE
# ---------------------------------------------------------------

def plot_ssvqe_convergence(molecule: str, E0_list, E1_list, optimizer_name="Adam", ansatz="UCCSD"):
    mol_norm = normalize_molecule_name(molecule)

    plt.figure(figsize=(8, 6))
    plt.plot(range(len(E0_list)), E0_list, label="Ground (E₀)")
    plt.plot(range(len(E1_list)), E1_list, label="1st Excited (E₁)")

    title = build_title(molecule, f"SSVQE: Convergence ({ansatz}, {optimizer_name})")
    fname = build_filename("SSVQE_Convergence", mol_norm, ansatz, optimizer_name)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    _save(fname)
