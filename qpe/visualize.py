"""
Visualization utilities for QPE.
Generates probability histograms and noise/parameter sweep plots.
"""

import os
import matplotlib.pyplot as plt
from qpe.io_utils import IMG_DIR, ensure_dirs


# ---------------------------------------------------------------------
# Distribution Plot
# ---------------------------------------------------------------------
def plot_qpe_distribution(result: dict, show: bool = True, save: bool = True):
    """
    Bar plot of measured ancilla state probabilities.

    Args:
        result (dict): Output dictionary from run_qpe()
        show (bool): Whether to display the plot interactively
        save (bool): Whether to save the figure to IMG_DIR
    """
    probs = result.get("probs", {})
    items = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    xs = [f"|{b}⟩" for b, _ in items]
    ys = [p for _, p in items]

    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys, color="#007acc", edgecolor="black", alpha=0.8)
    plt.xlabel("Ancilla register state", fontsize=11)
    plt.ylabel("Probability", fontsize=11)
    plt.title(
        f"QPE Phase Distribution – {result.get('molecule', '?')}\n"
        f"(ancillas={result.get('n_ancilla')}, noise={result.get('noise')})",
        fontsize=12,
    )
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save figure
    if save:
        ensure_dirs()
        fname = os.path.join(
            IMG_DIR,
            f"{result.get('molecule', 'qpe')}_QPE_{result.get('n_ancilla')}q.png",
        )
        plt.savefig(fname, dpi=200)
        print(f"📸 Saved QPE distribution plot → {fname}")

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# Sweep Plot (Noise / Ancilla / Time)
# ---------------------------------------------------------------------
def plot_qpe_sweep(
    x_values,
    y_means,
    y_stds=None,
    x_label: str = "Parameter",
    y_label: str = "Energy (Ha)",
    title: str = "QPE Sweep",
    ref_value: float | None = None,
    ref_label: str = "Reference",
    save_name: str | None = None,
):
    """
    Plot mean ± std of QPE results over a swept parameter.

    Args:
        x_values: list of x-axis parameter values
        y_means: list of mean values
        y_stds: list of standard deviations (optional)
        x_label: axis label
        y_label: axis label
        title: plot title
        ref_value: optional horizontal reference line (e.g., Hartree–Fock energy)
        ref_label: label for reference line
        save_name: filename for saving (optional)
    """
    plt.figure(figsize=(6, 4))
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
        path = save_qpe_plot(save_name)
        plt.savefig(path, dpi=200)
        print(f"📊 Saved QPE sweep plot → {path}")

    plt.show()


# ---------------------------------------------------------------------
# Save Helper
# ---------------------------------------------------------------------
def save_qpe_plot(name: str) -> str:
    """Return a full image path in IMG_DIR and ensure directory exists."""
    ensure_dirs()
    path = os.path.join(IMG_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
