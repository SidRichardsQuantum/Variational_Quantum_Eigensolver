"""
qpe.visualize
==============
High-quality plotting utilities for Quantum Phase Estimation (QPE),
fully unified with the project's global plotting system in
`common.plotting`.

Provides:
    • plot_qpe_distribution  – histogram of ancilla measurement outcomes
    • plot_qpe_sweep         – generic sweep plotting (noise, ancillas, t, etc.)

All plots are saved via:
    common.plotting.build_filename
    common.plotting.save_plot

This guarantees:
    • uniform PNG naming across VQE + QPE
    • safe molecule names
    • plots stored in /plots/
    • consistent DPI / formatting for publication-quality figures
"""

from __future__ import annotations

from typing import Dict, Any, Sequence, Optional

import matplotlib.pyplot as plt

from vqe_qpe_common.plotting import (
    build_filename,
    save_plot,
    format_molecule_name,
)


# ---------------------------------------------------------------------
# QPE Probability Distribution Plot
# ---------------------------------------------------------------------
def plot_qpe_distribution(
    result: Dict[str, Any],
    *,
    show: bool = True,
    save: bool = True,
) -> None:
    """
    Plot the ancilla probability distribution from a QPE run.

    Parameters
    ----------
    result : dict
        QPE result dictionary produced by run_qpe().
    show : bool
        Display figure window.
    save : bool
        Save figure via common plotting system.
    """
    probs = result.get("probs", {})
    if not probs:
        print("⚠️ No probability data found in QPE result — skipping plot.")
        return

    molecule = format_molecule_name(result.get("molecule", "QPE"))
    n_anc = int(result.get("n_ancilla", 0))

    noise = result.get("noise", {})
    p_dep = float(noise.get("p_dep", 0.0))
    p_amp = float(noise.get("p_amp", 0.0))

    # Sort by probability descending
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    xs = [f"|{b}⟩" for b, _ in items]
    ys = [float(p) for _, p in items]

    # Figure
    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys, alpha=0.85, edgecolor="black")

    plt.xlabel("Ancilla State", fontsize=11)
    plt.ylabel("Probability", fontsize=11)

    noise_suffix = ""
    if p_dep > 0 or p_amp > 0:
        noise_suffix = f" • noise(p_dep={p_dep}, p_amp={p_amp})"

    plt.title(
        f"{molecule} QPE Distribution ({n_anc} ancilla){noise_suffix}",
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
            },
        )
        save_plot(fname)

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------
# Generic QPE Sweep Plot
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
    show: bool = True,
    save: bool = True,
) -> None:
    """
    A general-purpose plotting routine for QPE sweeps:
        • sweep over noise strengths
        • sweep over t parameter
        • sweep over number of ancilla qubits
        • sweep over geometry/bond length (rare for QPE, but possible)

    Parameters
    ----------
    x_values : list
        Parameter values for sweep
    y_means : list
        Mean energies or phases
    y_stds : list, optional
        Standard deviations
    molecule : str
        Molecule label
    sweep_label : str
        X-axis label
    ylabel : str
        Y-axis label (energy or phase)
    title : str
        Plot title
    ref_value : float, optional
        Add horizontal reference line
    ancilla : int, optional
        Ancilla count for filename metadata
    noise_params : dict, optional
        Noise info for filename metadata
    """

    molecule = format_molecule_name(molecule)
    p_dep = float((noise_params or {}).get("p_dep", 0.0))
    p_amp = float((noise_params or {}).get("p_amp", 0.0))

    plt.figure(figsize=(6.5, 4.5))

    if y_stds is not None:
        plt.errorbar(
            x_values,
            y_means,
            yerr=y_stds,
            fmt="o-",
            capsize=4,
            label="QPE mean ± std",
        )
    else:
        plt.plot(x_values, y_means, "o-", label="QPE mean")

    if ref_value is not None:
        plt.axhline(
            ref_value,
            linestyle="--",
            color="gray",
            label=ref_label,
            alpha=0.8,
        )

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
                "tag": title.replace(" ", "_").lower(),
            },
        )
        save_plot(fname)

    if show:
        plt.show()
    else:
        plt.close()
