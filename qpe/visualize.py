"""
qpe/visualize.py
================
Plotting utilities for Quantum Phase Estimation (QPE).

Provides:
  • Probability distribution plots of ancilla bitstrings
  • Sweep plots (noise, ancillas, time, etc.)
  
This version uses the unified project-wide plotting system defined in
`common.plotting`, ensuring full consistency with VQE plotting.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Dict, Any

from common.plotting import (
    build_filename,
    save_plot,
    format_molecule_name,
)


# ---------------------------------------------------------------------
# QPE Distribution Plot
# ---------------------------------------------------------------------
def plot_qpe_distribution(
    result: Dict[str, Any],
    show: bool = True,
    save: bool = True,
) -> None:
    """
    Plot the measured ancilla probability distribution from a QPE run.

    Parameters
    ----------
    result : dict
        Output dictionary produced by run_qpe().
    show : bool
        Whether to display the figure.
    save : bool
        Whether to save the figure through the unified plotting system.
    """

    probs: Dict[str, float] = result.get("probs", {})
    if not probs:
        print("⚠️ No probabilities in QPE result; plot skipped.")
        return

    molecule = format_molecule_name(result.get("molecule", "QPE"))
    n_anc = int(result.get("n_ancilla", 0))
    noise = result.get("noise", {})
    p_dep = noise.get("p_dep", 0.0)
    p_amp = noise.get("p_amp", 0.0)

    # Sort by probability (high → low)
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    xs = [f"|{b}⟩" for b, _ in items]
    ys = [p for _, p in items]

    # Figure
    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys, alpha=0.85, edgecolor="black")

    plt.xlabel("Ancilla State", fontsize=11)
    plt.ylabel("Probability", fontsize=11)

    noise_label = ""
    if p_dep > 0 or p_amp > 0:
        noise_label = f" • noise(p_dep={p_dep}, p_amp={p_amp})"

    plt.title(
        f"{molecule} QPE Distribution ({n_anc} ancilla){noise_label}",
        fontsize=12
    )

    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        fname = build_filename(
            molecule=molecule,
            topic="qpe_distribution",
            extras={
                "anc": n_anc,
                "pdep": p_dep,
                "pamp": p_amp,
            }
        )
        save_plot(fname)

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# Sweep Plot (Noise / Ancilla Count / Time Parameter)
# ---------------------------------------------------------------------
def plot_qpe_sweep(
    x_values: Sequence[float],
    y_means: Sequence[float],
    y_stds: Optional[Sequence[float]] = None,
    *,
    molecule: str = "?",
    sweep_label: str = "Sweep parameter",
    ylabel: str = "Energy (Ha)",
    title: str = "QPE Sweep",
    ref_value: Optional[float] = None,
    ref_label: str = "Reference",
    ancilla: Optional[int] = None,
    noise_params: Optional[Dict[str, float]] = None,
    save: bool = True,
    show: bool = True,
) -> None:
    """
    Plot a sweep of QPE-computed energies or phases (mean ± std).

    Parameters
    ----------
    x_values : list
        Parameter values (noise strengths, ancilla counts, times, etc.)
    y_means : list
        Mean measured energies or phases.
    y_stds : list, optional
        Standard deviations.
    molecule : str
        Molecule label.
    sweep_label : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Plot title.
    ref_value : float, optional
        Reference horizontal line (e.g. Hartree–Fock energy).
    ancilla : int, optional
        Ancilla count for filename metadata.
    noise_params : dict, optional
        Noise dict: {"p_dep": ..., "p_amp": ...}
    save : bool
        Whether to save the plot.
    show : bool
        Whether to display the plot.
    """

    molecule = format_molecule_name(molecule)
    p_dep = (noise_params or {}).get("p_dep", 0.0)
    p_amp = (noise_params or {}).get("p_amp", 0.0)

    plt.figure(figsize=(6.5, 4.5))

    if y_stds is not None:
        plt.errorbar(
            x_values,
            y_means,
            yerr=y_stds,
            fmt="o-",
            capsize=4,
            label="QPE mean ± std"
        )
    else:
        plt.plot(x_values, y_means, "o-", label="QPE mean")

    if ref_value is not None:
        plt.axhline(ref_value, linestyle="--", color="gray", label=ref_label)

    plt.xlabel(sweep_label)
    plt.ylabel(ylabel)
    plt.title(f"{molecule} – {title}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save:
        fname = build_filename(
            molecule=molecule,
            topic="qpe_sweep",
            extras={
                "anc": ancilla,
                "pdep": p_dep,
                "pamp": p_amp,
                "topic": title.replace(" ", "_").lower(),
            }
        )
        save_plot(fname)

    if show:
        plt.show()
    else:
        plt.close()
