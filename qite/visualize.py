"""
qite.visualize
--------------
Plotting utilities for VarQITE / QITE-style routines.

All PNG outputs are routed to:
    images/qite/<MOLECULE>/

Design notes
------------
- Reuses shared filename/title helpers from common.plotting.
- Provides a local save helper to route outputs to images/qite/.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt

from common.naming import format_token
from common.plotting import (
    build_filename,
    format_molecule_title,
    infer_noise_type,
    save_plot,
)


def _safe_title(*parts: object) -> str:
    items = [str(p) for p in parts if p is not None and str(p).strip() != ""]
    return " — ".join(items)


# -----------------------------------------------------------------------------
# Primary plots
# -----------------------------------------------------------------------------
def plot_convergence(
    energies: Sequence[float],
    *,
    molecule: str = "molecule",
    method: str = "VarQITE",
    ansatz: Optional[str] = None,
    step_label: str = "Iteration",
    ylabel: str = "Energy (Ha)",
    seed: Optional[int] = None,
    show: bool = True,
    save: bool = True,
    # Optional noise metadata for titles/filenames
    dep_prob: float = 0.0,
    amp_prob: float = 0.0,
    noise_type: Optional[str] = None,
) -> None:
    """
    Plot energy convergence vs iteration.

    Notes
    -----
    VarQITE runs are noiseless by design; noise metadata is supported only for
    consistency with shared filename conventions (e.g. post-evaluation plots).
    """
    mol_title = format_molecule_title(molecule)

    plt.figure(figsize=(8, 5))
    xs = range(len(energies))
    plt.plot(xs, [float(e) for e in energies], lw=2)

    title = _safe_title(
        mol_title,
        f"{str(method).strip()} Convergence",
        ansatz if ansatz else None,
        (
            f"noise(dep={float(dep_prob):g}, amp={float(amp_prob):g})"
            if (float(dep_prob) > 0.0 or float(amp_prob) > 0.0)
            else None
        ),
    )

    plt.title(title)
    plt.xlabel(step_label)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if not bool(save):
        if bool(show):
            plt.show()
        else:
            plt.close()
        return

    nt = noise_type if noise_type is not None else infer_noise_type(dep_prob, amp_prob)

    fname = build_filename(
        topic="convergence",
        ansatz=ansatz,
        dep=(float(dep_prob) if float(dep_prob) > 0.0 else None),
        amp=(float(amp_prob) if float(amp_prob) > 0.0 else None),
        noise_scan=False,
        noise_type=nt,
        seed=seed,
        multi_seed=False,
    )

    save_plot(fname, kind="qite", molecule=molecule, show=bool(show))


def plot_noise_statistics(
    noise_levels: Sequence[float],
    deltaE_mean: Sequence[float],
    deltaE_std: Optional[Sequence[float]] = None,
    fidelity_mean: Optional[Sequence[float]] = None,
    fidelity_std: Optional[Sequence[float]] = None,
    *,
    molecule: str = "molecule",
    method: str = "VarQITE",
    ansatz: Optional[str] = None,
    noise_type: str = "depolarizing",
    seed: Optional[int] = None,
    show: bool = True,
    save: bool = True,
) -> None:
    """
    Plot ΔE and (optionally) fidelity vs noise level.

    Intended for post-evaluation noise studies (e.g., evaluating converged VarQITE
    parameters under default.mixed).
    """
    mol_title = format_molecule_title(molecule)
    nt = str(noise_type).strip().lower()

    has_fid = fidelity_mean is not None

    if has_fid:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        ax1.set_title(
            _safe_title(
                mol_title,
                f"{str(method).strip()} Noise Impact — {nt}",
                ansatz if ansatz else None,
            )
        )

        if deltaE_std is not None:
            ax1.errorbar(
                noise_levels,
                deltaE_mean,
                yerr=deltaE_std,
                fmt="o-",
                capsize=4,
                label="ΔE (mean ± std)",
            )
        else:
            ax1.plot(noise_levels, deltaE_mean, "o-", label="ΔE (mean)")

        ax1.set_ylabel("ΔE (Ha)")
        ax1.grid(True, alpha=0.4)
        ax1.legend()

        if fidelity_std is not None:
            ax2.errorbar(
                noise_levels,
                fidelity_mean,
                yerr=fidelity_std,
                fmt="s-",
                capsize=4,
                label="Fidelity (mean ± std)",
            )
        else:
            ax2.plot(noise_levels, fidelity_mean, "s-", label="Fidelity (mean)")

        ax2.set_xlabel("Noise Probability")
        ax2.set_ylabel("Fidelity")
        ax2.grid(True, alpha=0.4)
        ax2.legend()

        plt.tight_layout()

    else:
        plt.figure(figsize=(8, 5))

        if deltaE_std is not None:
            plt.errorbar(
                noise_levels,
                deltaE_mean,
                yerr=deltaE_std,
                fmt="o-",
                capsize=4,
            )
        else:
            plt.plot(noise_levels, deltaE_mean, "o-")

        plt.title(
            _safe_title(
                mol_title,
                f"{str(method).strip()} ΔE vs Noise — {nt}",
                ansatz if ansatz else None,
            )
        )
        plt.xlabel("Noise Probability")
        plt.ylabel("ΔE (Ha)")
        plt.grid(True, alpha=0.4)
        plt.tight_layout()

    if not bool(save):
        if bool(show):
            plt.show()
        else:
            plt.close()
        return

    fname = build_filename(
        topic="noise_stats",
        ansatz=ansatz,
        noise_scan=True,
        noise_type=nt,
        seed=seed,
        multi_seed=True,
    )

    save_plot(fname, kind="qite", molecule=molecule, show=bool(show))


def plot_diagnostics(
    values: Sequence[float],
    *,
    molecule: str = "molecule",
    title: str = "Diagnostics",
    xlabel: str = "Iteration",
    ylabel: str = "Value",
    tag: str = "diagnostics",
    ansatz: Optional[str] = None,
    seed: Optional[int] = None,
    show: bool = True,
    save: bool = True,
) -> None:
    """
    Generic single-curve diagnostic plot (e.g., residual norms, step sizes, etc.).
    """
    mol_title = format_molecule_title(molecule)

    plt.figure(figsize=(8, 5))
    xs = range(len(values))
    plt.plot(xs, [float(v) for v in values], lw=2)

    plt.title(_safe_title(mol_title, str(title).strip(), ansatz if ansatz else None))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()

    if not bool(save):
        if bool(show):
            plt.show()
        else:
            plt.close()
        return

    fname = build_filename(
        topic=format_token(tag),
        ansatz=ansatz,
        seed=seed,
        multi_seed=False,
    )

    save_plot(fname, kind="qite", molecule=molecule, show=bool(show))
