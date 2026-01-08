"""
vqe.visualize
-------------
Plotting utilities for VQE, SSVQE, and VQD.

Notes on multi-state plotting
-----------------------------
For SSVQE/VQD, trajectories are often indexed by *reference state* (SSVQE) or
*deflation step* (VQD). These curves can cross, and “state swapping” can occur
in terms of which curve ends at the lowest final energy.

Therefore:
- This module does NOT attempt to reorder trajectories automatically.
- If you want consistent “E0/E1/…” semantics in plots, pass an explicit
  `state_labels=` (or `order=`) based on your own policy (e.g. final-energy sort).
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt

from vqe_qpe_common.plotting import build_filename, save_plot


def _safe_title(*parts):
    return " — ".join([str(p) for p in parts if p is not None])


def plot_convergence(
    energies_noiseless,
    molecule: str,
    energies_noisy=None,
    optimizer: str = "Adam",
    ansatz: str = "UCCSD",
    dep_prob: float = 0.0,
    amp_prob: float = 0.0,
    seed: int | None = None,
    show: bool = True,
):
    plt.figure(figsize=(8, 5))
    steps = range(len(energies_noiseless))
    plt.plot(steps, energies_noiseless, label="Noiseless", lw=2)

    noisy = energies_noisy is not None
    if noisy:
        plt.plot(
            range(len(energies_noisy)),
            energies_noisy,
            label="Noisy",
            lw=2,
            linestyle="--",
        )

    if noisy:
        title = _safe_title(
            f"{molecule}",
            f"VQE Convergence ({optimizer}, {ansatz})",
            f"Noise: dep={dep_prob}, amp={amp_prob}",
        )
    else:
        title = _safe_title(f"{molecule}", f"VQE Convergence ({optimizer}, {ansatz})")

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

    fname = build_filename(
        topic="convergence",
        ansatz=ansatz,
        optimizer=optimizer,
        seed=seed,
        dep=dep_prob if noisy else None,
        amp=amp_prob if noisy else None,
        noise_scan=False,
        multi_seed=False,
    )
    save_plot(fname, kind="vqe", molecule=molecule, show=show)


def plot_optimizer_comparison(
    molecule: str,
    results: dict,
    ansatz: str = "UCCSD",
    seed: int | None = None,
    show: bool = True,
):
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

    fname = build_filename(
        topic="optimizer_conv",
        ansatz=ansatz,
        seed=seed,
        multi_seed=False,
    )
    save_plot(fname, kind="vqe", molecule=molecule, show=show)


def plot_ansatz_comparison(
    molecule: str,
    results: dict,
    optimizer: str = "Adam",
    seed: int | None = None,
    show: bool = True,
):
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

    fname = build_filename(
        topic="ansatz_conv",
        optimizer=optimizer,
        seed=seed,
        multi_seed=False,
    )
    save_plot(fname, kind="vqe", molecule=molecule, show=show)


def plot_noise_statistics(
    molecule: str,
    noise_levels,
    energy_means,
    energy_stds,
    fidelity_means,
    fidelity_stds,
    optimizer_name: str = "Adam",
    ansatz_name: str = "UCCSD",
    noise_type: str = "Depolarizing",
    show: bool = True,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.errorbar(noise_levels, energy_means, yerr=energy_stds, fmt="o-", capsize=4)
    ax1.set_ylabel("ΔE (Ha)")
    ax1.set_title(
        _safe_title(
            molecule,
            f"VQE Noise Impact — {noise_type}",
            f"{optimizer_name}, {ansatz_name}",
        )
    )
    ax1.grid(True, alpha=0.4)

    ax2.errorbar(noise_levels, fidelity_means, yerr=fidelity_stds, fmt="s-", capsize=4)
    ax2.set_xlabel("Noise Probability")
    ax2.set_ylabel("Fidelity")
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()

    nt = str(noise_type).strip().lower()
    if nt.startswith("dep"):
        nt = "depolarizing"
    elif nt.startswith("amp"):
        nt = "amplitude"
    elif nt.startswith("comb"):
        nt = "combined"

    fname = build_filename(
        topic="noise_stats",
        ansatz=ansatz_name,
        optimizer=optimizer_name,
        noise_scan=True,
        noise_type=nt,
        multi_seed=True,
    )
    save_plot(fname, kind="vqe", molecule=molecule, show=show)


def plot_multi_state_convergence(
    energies_per_state=None,
    ssvqe_or_vqd: str = "SSVQE",
    *,
    molecule: str = "molecule",
    ansatz: str = "UCCSD",
    optimizer: str = "Adam",
    optimizer_name=None,
    E0_list=None,
    E1_list=None,
    seed: int | None = None,
    show: bool = True,
    save: bool = True,
    # New (backwards compatible) controls:
    state_labels: Optional[Sequence[str]] = None,
    order: Optional[Sequence[int]] = None,
):
    """
    Plot multi-state convergence trajectories.

    Parameters
    ----------
    energies_per_state
        Either a list-of-lists (index semantics defined by the caller) or a dict.
        If dict, keys are sorted and values plotted in that order.
    state_labels
        Optional labels for each plotted curve. If omitted, uses "State i".
    order
        Optional permutation to reorder curves BEFORE plotting.
        Useful if you want to plot in a canonical order (e.g., final-energy-sorted),
        while keeping the underlying data unchanged.
        Example: order=[1,0] will plot trajectory[1] as "State 0" (or label 0).
    """
    if optimizer_name is not None:
        optimizer = optimizer_name

    if energies_per_state is None:
        if E0_list is None:
            raise TypeError("Provide energies_per_state or (E0_list, E1_list).")
        energies_per_state = [E0_list] if E1_list is None else [E0_list, E1_list]

    if isinstance(energies_per_state, dict):
        trajectories = [
            energies_per_state[k] for k in sorted(energies_per_state.keys())
        ]
    else:
        trajectories = list(energies_per_state)

    if order is not None:
        order = [int(i) for i in order]
        if sorted(order) != list(range(len(trajectories))):
            raise ValueError(
                f"order must be a permutation of 0..{len(trajectories)-1}, got {order}"
            )
        trajectories = [trajectories[i] for i in order]
        if state_labels is not None:
            state_labels = [state_labels[i] for i in order]

    n_states = len(trajectories)

    if state_labels is not None and len(state_labels) != n_states:
        raise ValueError(
            f"state_labels must have length {n_states}, got {len(state_labels)}"
        )

    plt.figure(figsize=(7, 4.5))
    for i, E_list in enumerate(trajectories):
        lbl = state_labels[i] if state_labels is not None else f"State {i}"
        plt.plot(E_list, label=lbl)

    tag = str(ssvqe_or_vqd).strip().upper()
    if tag == "SSVQE":
        method_name = "SSVQE"
    elif tag == "VQD":
        method_name = "VQD"
    else:
        method_name = tag

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} {method_name} ({n_states} states) – {ansatz}, {optimizer}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        fname = build_filename(
            topic=f"{method_name.lower()}_conv",
            ansatz=ansatz,
            optimizer=optimizer,
            seed=seed,
            multi_seed=False,
        )
        save_plot(fname, kind="vqe", molecule=molecule, show=show)
    elif show:
        plt.show()
    else:
        plt.close()
