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

from common.plotting import build_filename, format_molecule_title, save_plot


def _safe_title(*parts):
    return " — ".join([str(p) for p in parts if p is not None])


def plot_lr_vqe_spectrum(
    exact_evals,
    lr_result: dict,
    *,
    molecule_label: str,
    show: bool = True,
    save: bool = False,
    title: str | None = None,
):
    import matplotlib.pyplot as _plt
    import numpy as _np

    exact_evals = _np.asarray(exact_evals, dtype=float).ravel()
    E0 = float(lr_result["reference_energy"])
    lr_energies = _np.asarray(lr_result["eigenvalues"], dtype=float).ravel()

    ref = (lr_result.get("diagnostics", {}) or {}).get("reference", {}) or {}
    ansatz = ref.get("ansatz", None)
    optimizer = ref.get("optimizer", None)
    seed = ref.get("seed", None)

    nearest_idx, nearest_E, nearest_err = [], [], []
    for E in lr_energies:
        diffs = _np.abs(exact_evals - float(E))
        j = int(_np.argmin(diffs))
        nearest_idx.append(j)
        nearest_E.append(float(exact_evals[j]))
        nearest_err.append(float(diffs[j]))

    nearest_idx = _np.asarray(nearest_idx, dtype=int)
    nearest_E = _np.asarray(nearest_E, dtype=float)
    nearest_err = _np.asarray(nearest_err, dtype=float)

    max_j = int(nearest_idx.max()) if nearest_idx.size else 0
    m_show = int(max(10, max_j + 6))
    exact_show = _np.asarray(exact_evals[:m_show], dtype=float)

    _plt.figure(figsize=(10, 4.2))

    x_exact = _np.arange(exact_show.size)
    _plt.vlines(
        x_exact,
        ymin=float(exact_show.min()),
        ymax=exact_show,
        linewidth=2,
        label="Exact eigenvalues",
    )

    _plt.axhline(E0, linestyle="--", linewidth=1.5, label="VQE reference E0")

    if lr_energies.size:
        x_lr = nearest_idx.astype(float)
        _plt.scatter(
            x_lr,
            lr_energies,
            s=70,
            marker="x",
            color="red",
            label="LR-VQE: E0 + ω",
        )

        for x, E_lr, E_ex in zip(x_lr, lr_energies, nearest_E):
            _plt.plot([x, x], [E_ex, E_lr], linewidth=1)

        for x, E_lr, dE in zip(x_lr, lr_energies, nearest_err):
            _plt.annotate(
                f"|ΔE|={float(dE):.1e}",
                (float(x), float(E_lr)),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    molecule_title = format_molecule_title(molecule_label)

    if title is None:
        suffix = None
        if ansatz is not None or optimizer is not None:
            suffix = f"{optimizer}, {ansatz}" if optimizer is not None else f"{ansatz}"
        title = _safe_title(f"{molecule_title} spectrum: exact vs LR-VQE", suffix)

    _plt.title(title)
    _plt.xlabel("Eigenvalue level index")
    _plt.ylabel("Energy (Ha)")
    _plt.grid(True, alpha=0.3)
    _plt.xlim(-0.8, exact_show.size - 0.2)
    _plt.legend()
    _plt.tight_layout()

    if save:
        fname = build_filename(
            topic="lr_vqe_spectrum",
            ansatz=ansatz,
            optimizer=optimizer,
            seed=seed,
            multi_seed=False,
        )
        save_plot(fname, kind="vqe", molecule=molecule_label, show=(show or save))

    if show or save:
        _plt.show()
    else:
        _plt.close()


def plot_eom_vqe_spectrum(
    exact_evals,
    eom_result: dict,
    *,
    molecule_label: str,
    show: bool = True,
    save: bool = False,
    title: str | None = None,
):
    import matplotlib.pyplot as _plt
    import numpy as _np

    exact_evals = _np.asarray(exact_evals, dtype=float).ravel()
    E0 = float(eom_result["reference_energy"])
    eom_energies = _np.asarray(eom_result["eigenvalues"], dtype=float).ravel()

    ref = (eom_result.get("diagnostics", {}) or {}).get("reference", {}) or {}
    ansatz = ref.get("ansatz", None)
    optimizer = ref.get("optimizer", None)
    seed = ref.get("seed", None)

    nearest_idx, nearest_E, nearest_err = [], [], []
    for E in eom_energies:
        diffs = _np.abs(exact_evals - float(E))
        j = int(_np.argmin(diffs))
        nearest_idx.append(j)
        nearest_E.append(float(exact_evals[j]))
        nearest_err.append(float(diffs[j]))

    nearest_idx = _np.asarray(nearest_idx, dtype=int)
    nearest_E = _np.asarray(nearest_E, dtype=float)
    nearest_err = _np.asarray(nearest_err, dtype=float)

    max_j = int(nearest_idx.max()) if nearest_idx.size else 0
    m_show = int(max(10, max_j + 6))
    exact_show = _np.asarray(exact_evals[:m_show], dtype=float)

    _plt.figure(figsize=(10, 4.2))

    x_exact = _np.arange(exact_show.size)
    _plt.vlines(
        x_exact,
        ymin=float(exact_show.min()),
        ymax=exact_show,
        linewidth=2,
        label="Exact eigenvalues",
    )

    _plt.axhline(E0, linestyle="--", linewidth=1.5, label="VQE reference E0")

    if eom_energies.size:
        x_eom = nearest_idx.astype(float)
        _plt.scatter(
            x_eom,
            eom_energies,
            s=70,
            marker="x",
            color="red",
            label="EOM-VQE: E0 + ω",
        )

        for x, E_eom, E_ex in zip(x_eom, eom_energies, nearest_E):
            _plt.plot([x, x], [E_ex, E_eom], linewidth=1)

        for x, E_eom, dE in zip(x_eom, eom_energies, nearest_err):
            _plt.annotate(
                f"|ΔE|={float(dE):.1e}",
                (float(x), float(E_eom)),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    molecule_title = format_molecule_title(molecule_label)

    if title is None:
        suffix = None
        if ansatz is not None or optimizer is not None:
            suffix = f"{optimizer}, {ansatz}" if optimizer is not None else f"{ansatz}"
        title = _safe_title(f"{molecule_title} spectrum: exact vs EOM-VQE", suffix)

    _plt.title(title)
    _plt.xlabel("Eigenvalue level index")
    _plt.ylabel("Energy (Ha)")
    _plt.grid(True, alpha=0.3)
    _plt.xlim(-0.8, exact_show.size - 0.2)
    _plt.legend()
    _plt.tight_layout()

    if save:
        fname = build_filename(
            topic="eom_vqe_spectrum",
            ansatz=ansatz,
            optimizer=optimizer,
            seed=seed,
            multi_seed=False,
        )
        save_plot(fname, kind="vqe", molecule=molecule_label, show=(show or save))

    if show or save:
        _plt.show()
    else:
        _plt.close()


def plot_eom_qse_spectrum(
    exact_evals,
    eom_qse_result: dict,
    *,
    molecule_label: str,
    show: bool = True,
    save: bool = False,
    title: str | None = None,
):
    import matplotlib.pyplot as _plt
    import numpy as _np

    exact_evals = _np.asarray(exact_evals, dtype=float).ravel()
    E0 = float(eom_qse_result["reference_energy"])
    eom_energies = _np.asarray(eom_qse_result["eigenvalues"], dtype=float).ravel()

    ref = (eom_qse_result.get("diagnostics", {}) or {}).get("reference", {}) or {}
    # EOM-QSE currently stores reference details in config, not diagnostics; keep consistent fallback.
    if not ref:
        cfg_ref = (eom_qse_result.get("config", {}) or {}).get(
            "vqe_reference", {}
        ) or {}
        ref = cfg_ref

    ansatz = ref.get("ansatz", None)
    optimizer = ref.get("optimizer", None)
    seed = ref.get("seed", None)

    nearest_idx, nearest_E, nearest_err = [], [], []
    for E in eom_energies:
        diffs = _np.abs(exact_evals - float(E))
        j = int(_np.argmin(diffs))
        nearest_idx.append(j)
        nearest_E.append(float(exact_evals[j]))
        nearest_err.append(float(diffs[j]))

    nearest_idx = _np.asarray(nearest_idx, dtype=int)
    nearest_E = _np.asarray(nearest_E, dtype=float)
    nearest_err = _np.asarray(nearest_err, dtype=float)

    max_j = int(nearest_idx.max()) if nearest_idx.size else 0
    m_show = int(max(10, max_j + 6))
    exact_show = _np.asarray(exact_evals[:m_show], dtype=float)

    _plt.figure(figsize=(10, 4.2))

    x_exact = _np.arange(exact_show.size)
    _plt.vlines(
        x_exact,
        ymin=float(exact_show.min()),
        ymax=exact_show,
        linewidth=2,
        label="Exact eigenvalues",
    )

    _plt.axhline(E0, linestyle="--", linewidth=1.5, label="VQE reference E0")

    if eom_energies.size:
        x_eom = nearest_idx.astype(float)
        _plt.scatter(
            x_eom,
            eom_energies,
            s=70,
            marker="x",
            color="red",
            label="EOM-QSE: E0 + ω",
        )

        for x, E_eom, E_ex in zip(x_eom, eom_energies, nearest_E):
            _plt.plot([x, x], [E_ex, E_eom], linewidth=1)

        for x, E_eom, dE in zip(x_eom, eom_energies, nearest_err):
            _plt.annotate(
                f"|ΔE|={float(dE):.1e}",
                (float(x), float(E_eom)),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    molecule_title = format_molecule_title(molecule_label)

    if title is None:
        suffix = None
        if ansatz is not None or optimizer is not None:
            suffix = f"{optimizer}, {ansatz}" if optimizer is not None else f"{ansatz}"
        title = _safe_title(f"{molecule_title} spectrum: exact vs EOM-QSE", suffix)

    _plt.title(title)
    _plt.xlabel("Eigenvalue level index")
    _plt.ylabel("Energy (Ha)")
    _plt.grid(True, alpha=0.3)
    _plt.xlim(-0.8, exact_show.size - 0.2)
    _plt.legend()
    _plt.tight_layout()

    if save:
        fname = build_filename(
            topic="eom_qse_spectrum",
            ansatz=ansatz,
            optimizer=optimizer,
            seed=seed,
            multi_seed=False,
        )
        save_plot(fname, kind="vqe", molecule=molecule_label, show=(show or save))

    if show or save:
        _plt.show()
    else:
        _plt.close()


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
    molecule_title = format_molecule_title(molecule)
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
            f"{molecule_title}",
            f"VQE Convergence ({optimizer}, {ansatz})",
            f"Noise: dep={dep_prob}, amp={amp_prob}",
        )
    else:
        title = _safe_title(
            f"{molecule_title}", f"VQE Convergence ({optimizer}, {ansatz})"
        )

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
    molecule_title = format_molecule_title(molecule)
    plt.figure(figsize=(8, 5))
    min_len = min(len(v) for v in results.values())

    for opt, energies in results.items():
        plt.plot(range(min_len), energies[:min_len], label=opt)

    plt.title(_safe_title(molecule_title, f"VQE Optimizer Comparison ({ansatz})"))
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
    molecule_title = format_molecule_title(molecule)
    plt.figure(figsize=(8, 5))
    min_len = min(len(v) for v in results.values())

    for ans, energies in results.items():
        plt.plot(range(min_len), energies[:min_len], label=ans)

    plt.title(_safe_title(molecule_title, f"VQE Ansatz Comparison ({optimizer})"))
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
    molecule_title = format_molecule_title(molecule)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.errorbar(noise_levels, energy_means, yerr=energy_stds, fmt="o-", capsize=4)
    ax1.set_ylabel("ΔE (Ha)")
    ax1.set_title(
        _safe_title(
            molecule_title,
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
    state_labels: Optional[Sequence[str]] = None,
    order: Optional[Sequence[int]] = None,
):
    molecule_title = format_molecule_title(molecule)

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
                f"order must be a permutation of 0..{len(trajectories) - 1}, got {order}"
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
    plt.title(
        f"{molecule_title} {method_name} ({n_states} states) – {ansatz}, {optimizer}"
    )
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
