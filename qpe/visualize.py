"""
qpe.visualize
-------------
Plotting utilities for Quantum Phase Estimation (QPE).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt

from vqe_qpe_common.plotting import build_filename, format_molecule_name, save_plot


def _infer_noise_type(p_dep: float, p_amp: float) -> Optional[str]:
    if p_dep > 0 and p_amp > 0:
        return "combined"
    if p_dep > 0 and p_amp == 0:
        return "depolarizing"
    if p_dep == 0 and p_amp > 0:
        return "amplitude"
    return None


def _extract_t(result: Dict[str, Any]) -> Optional[float]:
    for key in ("t", "time", "evolution_time"):
        if key in result and result[key] is not None:
            try:
                return float(result[key])
            except Exception:
                return None
    return None


def plot_qpe_distribution(
    result: Dict[str, Any],
    *,
    show: bool = True,
    save: bool = True,
) -> None:
    probs = result.get("probs", {})
    if not probs:
        print("⚠️ No probability data found in QPE result — skipping plot.")
        return

    molecule = format_molecule_name(result.get("molecule", "QPE"))
    n_anc = int(result.get("n_ancilla", 0))
    t_val = _extract_t(result)

    noise = result.get("noise", {}) or {}
    p_dep = float(noise.get("p_dep", 0.0))
    p_amp = float(noise.get("p_amp", 0.0))

    items = sorted(probs.items(), key=lambda kv: -kv[1])
    xs = [f"|{b}⟩" for b, _ in items]
    ys = [float(p) for _, p in items]

    plt.figure(figsize=(8, 4))
    plt.bar(xs, ys, alpha=0.85, edgecolor="black")
    plt.xlabel("Ancilla State", fontsize=11)
    plt.ylabel("Probability", fontsize=11)

    noise_suffix = ""
    if p_dep > 0 or p_amp > 0:
        noise_suffix = f" • noise(p_dep={p_dep}, p_amp={p_amp})"

    t_suffix = ""
    if t_val is not None:
        t_suffix = f" • t={t_val}"

    plt.title(
        f"{molecule} QPE Distribution ({n_anc} ancilla){t_suffix}{noise_suffix}",
        fontsize=12,
    )
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save:
        fname = build_filename(
            topic="distribution",
            ancilla=n_anc if n_anc > 0 else None,
            t=t_val,
            dep=p_dep if p_dep > 0 else None,
            amp=p_amp if p_amp > 0 else None,
            noise_scan=False,
            noise_type=_infer_noise_type(p_dep, p_amp),
            multi_seed=False,
        )
        save_plot(fname, kind="qpe", molecule=molecule, show=show)
    elif show:
        plt.show()
    else:
        plt.close()


def plot_qpe_sweep(
    x_values: Sequence[float],
    y_means: Sequence[float],
    y_stds: Optional[Sequence[float]] = None,
    *,
    molecule: str = "?",
    sweep_label: str = "Sweep parameter",
    ylabel: str = "Energy (Ha)",
    title: str = "Sweep",
    ref_value: Optional[float] = None,
    ref_label: str = "Reference",
    ancilla: Optional[int] = None,
    t: Optional[float] = None,
    noise_params: Optional[Dict[str, float]] = None,
    show: bool = True,
    save: bool = True,
) -> None:
    molecule = format_molecule_name(molecule)
    noise_params = noise_params or {}
    p_dep = float(noise_params.get("p_dep", 0.0))
    p_amp = float(noise_params.get("p_amp", 0.0))

    plt.figure(figsize=(6.5, 4.5))

    if y_stds is not None:
        plt.errorbar(
            x_values,
            y_means,
            yerr=y_stds,
            fmt="o-",
            capsize=4,
            label="Mean ± std",
        )
    else:
        plt.plot(x_values, y_means, "o-", label="Mean")

    if ref_value is not None:
        plt.axhline(ref_value, linestyle="--", color="gray", label=ref_label, alpha=0.8)

    plt.xlabel(sweep_label)
    plt.ylabel(ylabel)
    plt.title(f"{molecule} – {title}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    noise_params = noise_params or {}
    p_dep = float(noise_params.get("p_dep", 0.0))
    p_amp = float(noise_params.get("p_amp", 0.0))

    is_noise_scan = (p_dep > 0.0) or (p_amp > 0.0)
    noise_type = _infer_noise_type(p_dep, p_amp) if is_noise_scan else None

    if save:
        fname = build_filename(
            topic="noise_scan" if is_noise_scan else "sweep",
            ancilla=ancilla,
            t=t,
            dep=p_dep if p_dep > 0 else None,
            amp=p_amp if p_amp > 0 else None,
            noise_scan=is_noise_scan,
            noise_type=noise_type,
            tag=(title.lower().replace(" ", "_").replace("(", "").replace(")", "")),
            multi_seed=False,
        )
        save_plot(fname, kind="qpe", molecule=molecule, show=show)
    elif show:
        plt.show()
    else:
        plt.close()
