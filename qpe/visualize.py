"""
qpe/visualize.py
================
Visualization utilities for Quantum Phase Estimation (QPE).

Provides:
  • Distribution plots of measured ancilla probabilities
  • Parameter / noise sweep plots with mean ± std
"""

from __future__ import annotations
import os
import matplotlib.pyplot as plt
from typing import Sequence, Optional
from qpe.io_utils import IMG_DIR, ensure_dirs, save_qpe_plot


# ---------------------------------------------------------------------
# Distribution Plot
# ---------------------------------------------------------------------
def plot_qpe_distribution(
    result: dict,
    show: bool = True,
    save: bool = True,
) -> None:
    """Plot measured ancilla probability distribution from QPE results.

    Parameters
    ----------
    result : dict
        Output dictionary from ``run_qpe()``.
    show : bool, optional
        Whether to display the figure interactively (default = True).
    save : bool, optional
        Whether to save the figure under ``qpe/images/`` (default = True).
    """
    probs = result.get("probs", {})
    if not probs:
        print("⚠️ No probabilities found in QPE result; skipping plot.")
        return

    items = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    xs = [f"|{b}⟩" for b, _ in items]
    ys = [p for _, p in items]

    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys, color="#007acc", edgecolor="black", alpha=0.85)
    plt.xlabel("Ancilla Register State", fontsize=11)
    plt.ylabel("Probability", fontsize=11)
    plt.title(
        f"QPE Phase Distribution – {result.get('molecule', '?')}\n"
        f"(ancillas={result.get('n_ancilla')}, noise={result.get('noise', {})})",
        fontsize=12,
    )
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save figure
    if save:
        ensure_dirs()
        safe_name = str(result.get("molecule", "qpe")).replace("+", "plus").replace(" ", "_")
        fname = f"{safe_name}_QPE_{result.get('n_ancilla', 0)}q.png"
        path = os.path.join(IMG_DIR, fname)
        plt.savefig(path, dpi=300)
        print(f"📸 Saved QPE distribution plot → {path}")

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# Sweep Plot (Noise / Ancilla / Time)
# ---------------------------------------------------------------------
def plot_qpe_sweep(
    x_values: Sequence[float],
    y_means: Sequence[float],
    y_stds: Optional[Sequence[float]] = None,
    x_label: str = "Parameter",
    y_label: str = "Energy (Ha)",
    title: str = "QPE Sweep",
    ref_value: Optional[float] = None,
    ref_label: str = "Reference",
    save_name: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot mean ± std of QPE results across a swept parameter.

    Parameters
    ----------
    x_values : list or array
        Parameter values on the x-axis (e.g., noise strength).
    y_means : list or array
        Mean measured energies or phases.
    y_stds : list or array, optional
        Standard deviations of the measured quantity.
    x_label, y_label : str, optional
        Axis labels.
    title : str, optional
        Plot title.
    ref_value : float, optional
        Optional horizontal reference line (e.g. Hartree–Fock energy).
    ref_label : str, optional
        Label for reference line.
    save_name : str, optional
        Filename for saving (e.g., ``"H2_QPE_noise_sweep.png"``).
    show : bool, optional
        Whether to display the plot interactively (default = True).
    """
    plt.figure(figsize=(6.5, 4.5))

    if y_stds is not None:
        plt.errorbar(
            x_values, y_means, yerr=y_stds, fmt="o-", capsize=3, label="QPE (mean ± std)"
        )
    else:
        plt.plot(x_values, y_means, "o-", label="QPE mean")

    if ref_value is not None:
        plt.axhline(ref_value, linestyle="--", color="gray", label=ref_label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_name:
        ensure_dirs()
        path = save_qpe_plot(save_name)
        plt.savefig(path, dpi=300)
        print(f"📊 Saved QPE sweep plot → {path}")

    if show:
        plt.show()
    else:
        plt.close()
