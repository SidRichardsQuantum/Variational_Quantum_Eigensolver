"""
vqe.core
--------
High-level orchestration of Variational Quantum Eigensolver (VQE) workflows.

Includes:
- Main VQE runner (`run_vqe`)
- Noise studies and multi-seed averaging
- Optimizer / ansatz comparisons
- Geometry scans (bond lengths, angles)
- Fermion-to-qubit mapping comparisons
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np

from common.problem import resolve_problem
from common.units import coordinate_unit_label

from .engine import (
    build_ansatz as engine_build_ansatz,
)
from .engine import (
    build_optimizer as engine_build_optimizer,
)
from .engine import (
    make_device,
    make_energy_qnode,
    make_state_qnode,
)
from .hamiltonian import build_hamiltonian, generate_geometry
from .io_utils import (
    ensure_dirs,
    load_run_record,
    make_filename_prefix,
    make_run_config_dict,
    run_signature,
    save_run_record,
)
from .optimizer import get_optimizer_stepsize
from .visualize import (
    plot_convergence,
    plot_noise_statistics,
)


# ================================================================
# SHARED HELPERS
# ================================================================
def compute_fidelity(pure_state, state_or_rho):
    """
    Fidelity between a pure state |ψ⟩ and either:
        - a statevector |φ⟩
        - or a density matrix ρ

    Returns |⟨ψ|φ⟩|² or ⟨ψ|ρ|ψ⟩ respectively.
    """
    state_or_rho = np.array(state_or_rho)
    pure_state = np.array(pure_state)

    if state_or_rho.ndim == 1:
        return float(abs(np.vdot(pure_state, state_or_rho)) ** 2)
    elif state_or_rho.ndim == 2:
        return float(np.real(np.vdot(pure_state, state_or_rho @ pure_state)))

    raise ValueError("Invalid state shape for fidelity computation")


def _noise_probs_for_type(noise_type: str, level: float) -> dict[str, float]:
    level_f = float(level)
    noise_type_norm = str(noise_type).strip().lower()

    out = {
        "depolarizing_prob": 0.0,
        "amplitude_damping_prob": 0.0,
        "phase_damping_prob": 0.0,
        "bit_flip_prob": 0.0,
        "phase_flip_prob": 0.0,
    }

    if noise_type_norm == "depolarizing":
        out["depolarizing_prob"] = level_f
    elif noise_type_norm in {"amplitude", "amplitude_damping"}:
        out["amplitude_damping_prob"] = level_f
    elif noise_type_norm in {"phase", "phase_damping"}:
        out["phase_damping_prob"] = level_f
    elif noise_type_norm in {"bit", "bit_flip"}:
        out["bit_flip_prob"] = level_f
    elif noise_type_norm in {"phase_flip", "phaseflip"}:
        out["phase_flip_prob"] = level_f
    elif noise_type_norm == "combined":
        out["depolarizing_prob"] = level_f
        out["amplitude_damping_prob"] = level_f
    else:
        raise ValueError(
            "Unknown noise_type "
            f"{noise_type!r} (use depolarizing, amplitude_damping, phase_damping, "
            "bit_flip, phase_flip, or combined)."
        )

    return out


def _format_noise_point(noise_kwargs: dict[str, float]) -> str:
    parts: list[str] = []
    mapping = (
        ("depolarizing_prob", "dep"),
        ("amplitude_damping_prob", "amp"),
        ("phase_damping_prob", "phase"),
        ("bit_flip_prob", "bit"),
        ("phase_flip_prob", "phase_flip"),
    )
    for key, label in mapping:
        val = float(noise_kwargs.get(key, 0.0))
        if val > 0.0:
            parts.append(f"{label}={val:g}")
    return ", ".join(parts)


# ================================================================
# MAIN VQE EXECUTION
# ================================================================
def run_vqe(
    molecule: str = "H2",
    seed: int = 0,
    steps: int = 75,
    stepsize: float | None = None,
    plot: bool = True,
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    phase_damping_prob: float = 0.0,
    bit_flip_prob: float = 0.0,
    phase_flip_prob: float = 0.0,
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    charge: int = 0,
    unit: str = "angstrom",
    mapping: str = "jordan_wigner",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    hamiltonian: qml.Hamiltonian | None = None,
    num_qubits: int | None = None,
    reference_state=None,
):
    ensure_dirs()
    np.random.seed(int(seed))
    resolved_stepsize = (
        get_optimizer_stepsize(str(optimizer_name))
        if stepsize is None
        else float(stepsize)
    )

    mapping_norm = str(mapping).strip().lower()
    problem = resolve_problem(
        molecule=molecule,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        charge=charge,
        mapping=mapping,
        unit=unit,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian=hamiltonian,
        num_qubits=num_qubits,
        reference_state=reference_state,
        reference_name="reference_state",
    )
    H = problem.hamiltonian
    qubits = problem.num_qubits
    molecule_label = problem.molecule_label
    symbols_out = problem.symbols
    coordinates_out = problem.coordinates
    basis_out = problem.basis
    charge_out = problem.charge
    unit_out = problem.unit
    resolved_active_electrons = problem.active_electrons
    resolved_active_orbitals = problem.active_orbitals
    cache_enabled = problem.cacheable

    # --- Configuration & caching ---
    cfg = make_run_config_dict(
        symbols=symbols_out,
        coordinates=coordinates_out,
        basis=basis_out,
        ansatz_desc=str(ansatz_name),
        optimizer_name=str(optimizer_name),
        stepsize=resolved_stepsize,
        max_iterations=int(steps),
        seed=int(seed),
        mapping=mapping_norm,
        noisy=bool(noisy),
        depolarizing_prob=float(depolarizing_prob),
        amplitude_damping_prob=float(amplitude_damping_prob),
        phase_damping_prob=float(phase_damping_prob),
        bit_flip_prob=float(bit_flip_prob),
        phase_flip_prob=float(phase_flip_prob),
        molecule_label=molecule_label,
        charge=charge_out,
        unit=unit_out,
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
    )

    prefix = None
    if cache_enabled:
        sig = run_signature(cfg)
        prefix = make_filename_prefix(
            cfg,
            noisy=bool(cfg.get("noise")),
            seed=int(seed),
            hash_str=sig,
            algo="vqe",
        )

        if not force:
            record = load_run_record(prefix)
            if record is not None:
                return record["result"]

    # --- Device, ansatz, optim, QNodes ---
    dev = make_device(int(qubits), noisy=bool(cfg.get("noise")))

    ansatz_fn, params0 = engine_build_ansatz(
        str(ansatz_name),
        int(qubits),
        seed=int(seed),
        symbols=symbols_out,
        coordinates=coordinates_out,
        charge=charge_out,
        basis=basis_out,
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        reference_state=reference_state,
    )

    energy_qnode = make_energy_qnode(
        H,
        dev,
        ansatz_fn,
        int(qubits),
        noisy=bool(cfg.get("noise")),
        depolarizing_prob=float(depolarizing_prob),
        amplitude_damping_prob=float(amplitude_damping_prob),
        phase_damping_prob=float(phase_damping_prob),
        bit_flip_prob=float(bit_flip_prob),
        phase_flip_prob=float(phase_flip_prob),
        symbols=symbols_out,
        coordinates=coordinates_out,
        charge=charge_out,
        basis=basis_out,
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        reference_state=reference_state,
    )

    state_qnode = make_state_qnode(
        dev,
        ansatz_fn,
        int(qubits),
        noisy=bool(cfg.get("noise")),
        depolarizing_prob=float(depolarizing_prob),
        amplitude_damping_prob=float(amplitude_damping_prob),
        phase_damping_prob=float(phase_damping_prob),
        bit_flip_prob=float(bit_flip_prob),
        phase_flip_prob=float(phase_flip_prob),
        symbols=symbols_out,
        coordinates=coordinates_out,
        charge=charge_out,
        basis=basis_out,
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
    )

    opt = engine_build_optimizer(str(optimizer_name), stepsize=resolved_stepsize)

    # --- Optimization loop ---
    params = np.array(params0, requires_grad=True)
    energies: list[float] = [float(energy_qnode(params))]

    params_history: list[list[float]] = [
        [float(x) for x in np.asarray(params, dtype=float).ravel()]
    ]

    for step in range(int(steps)):
        try:
            params, cost = opt.step_and_cost(energy_qnode, params)
            e = float(cost)
        except AttributeError:
            params = opt.step(energy_qnode, params)
            e = float(energy_qnode(params))

        energies.append(float(e))
        params_history.append(
            [float(x) for x in np.asarray(params, dtype=float).ravel()]
        )

        print(f"Step {step + 1:02d}/{steps}: E = {float(e):.6f} Ha")

    final_energy = float(energies[-1])
    final_params = params_history[-1]

    final_state = state_qnode(params)

    # --- Optional plot ---
    if plot:
        plot_convergence(
            energies,
            molecule_label,
            optimizer=str(optimizer_name),
            ansatz=str(ansatz_name),
            dep_prob=float(depolarizing_prob),
            amp_prob=float(amplitude_damping_prob),
            phase_prob=float(phase_damping_prob),
            bit_flip_prob=float(bit_flip_prob),
            phase_flip_prob=float(phase_flip_prob),
        )

    # --- Save ---
    result = {
        "energy": float(final_energy),
        "energies": [float(e) for e in energies],
        "steps": int(steps),
        "final_state_real": np.real(final_state).tolist(),
        "final_state_imag": np.imag(final_state).tolist(),
        "num_qubits": int(qubits),
        "active_electrons": resolved_active_electrons,
        "active_orbitals": resolved_active_orbitals,
        "final_params": final_params,
        "params_history": params_history,
    }

    if cache_enabled and prefix is not None:
        save_run_record(prefix, {"config": cfg, "result": result})
        print(f"\n💾 Saved run record: {prefix}.json\n")

    return result


# ================================================================
# OPTIMIZER COMPARISON
# ================================================================
def run_vqe_optimizer_comparison(
    molecule: str = "H2",
    ansatz_name: str = "RY-CZ",
    optimizers=None,
    steps: int = 50,
    stepsize=None,
    noisy: bool = True,
    depolarizing_prob: float = 0.05,
    amplitude_damping_prob: float = 0.05,
    phase_damping_prob: float = 0.0,
    bit_flip_prob: float = 0.0,
    phase_flip_prob: float = 0.0,
    seed: int = 0,
    mode: str = "convergence",
    noise_type: str = "depolarizing",
    noise_levels=None,
    seeds=None,
    reference: str = "per_seed_noiseless",
    force: bool = False,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    show: bool = True,
    plot: bool = True,
):
    """
    Compare classical optimizers for a fixed VQE instance.

    This function supports two modes:

    1) mode="convergence"
       - Runs each optimizer once (single seed, single noise point).
       - Returns energy trajectories vs iteration.

    2) mode="noise_stats" (new; for Noisy_Optimizer_Comparison)
       - Sweeps noise_levels and averages over seeds for each optimizer.
       - Computes:
           ΔE = E_noisy - E_ref   (reference from noiseless runs)
           Fidelity vs noiseless final state
         and returns mean/std vs noise level per optimizer.

    Parameters
    ----------
    stepsize : float | dict | None
        If omitted, use the calibrated default stepsize for each optimizer.
    noise_type : str
        Built-in channel to sweep in `mode="noise_stats"`.
        Supported values:
        `depolarizing`, `amplitude_damping`, `phase_damping`,
        `bit_flip`, `phase_flip`, `combined`.
    reference : str
        Currently only "per_seed_noiseless" is supported:
        compute noiseless reference energy/state for each seed (and optimizer).
    """
    import matplotlib.pyplot as plt

    from common.plotting import build_filename, save_plot

    optimizers = optimizers or ["Adam", "GradientDescent", "Momentum"]

    # -----------------------------
    # Helper: resolve stepsize
    # -----------------------------
    def _stepsize_for(opt_name: str) -> float:
        if stepsize is None:
            return get_optimizer_stepsize(opt_name)
        if isinstance(stepsize, dict):
            if opt_name not in stepsize:
                raise ValueError(
                    f"stepsize dict missing entry for optimizer '{opt_name}'. "
                    f"Provided keys: {list(stepsize.keys())}"
                )
            return float(stepsize[opt_name])
        return float(stepsize)

    # ============================================================
    # MODE 1: Legacy convergence comparison (single run per optimizer)
    # ============================================================
    if mode == "convergence":
        results = {}
        final_vals = {}

        for opt_name in optimizers:
            print(f"\n⚙️ Running optimizer: {opt_name}")
            res = run_vqe(
                molecule=molecule,
                steps=steps,
                stepsize=_stepsize_for(opt_name),
                plot=False,
                ansatz_name=ansatz_name,
                optimizer_name=opt_name,
                noisy=noisy,
                depolarizing_prob=depolarizing_prob,
                amplitude_damping_prob=amplitude_damping_prob,
                phase_damping_prob=phase_damping_prob,
                bit_flip_prob=bit_flip_prob,
                phase_flip_prob=phase_flip_prob,
                mapping=mapping,
                force=force,
                unit=unit,
                seed=int(seed),
            )
            results[opt_name] = res["energies"]
            final_vals[opt_name] = res["energy"]

        if plot:
            plt.figure(figsize=(8, 5))
            min_len = min(len(v) for v in results.values())
            for opt, energies in results.items():
                plt.plot(range(min_len), energies[:min_len], label=opt)

            title_noise = ""
            if noisy:
                noise_point = {
                    "depolarizing_prob": depolarizing_prob,
                    "amplitude_damping_prob": amplitude_damping_prob,
                    "phase_damping_prob": phase_damping_prob,
                    "bit_flip_prob": bit_flip_prob,
                    "phase_flip_prob": phase_flip_prob,
                }
                title_noise = f" ({_format_noise_point(noise_point)})"
            plt.title(f"{molecule} – Optimizer Comparison ({ansatz_name}){title_noise}")
            plt.xlabel("Iteration")
            plt.ylabel("Energy (Ha)")
            plt.grid(True, alpha=0.4)
            plt.legend()
            plt.tight_layout()

            multi = bool(seeds) and (len(seeds) > 1)

            fname = build_filename(
                topic="optimizer_conv",
                ansatz=ansatz_name,
                dep=depolarizing_prob if noisy else None,
                amp=amplitude_damping_prob if noisy else None,
                seed=None if multi else seed,
                multi_seed=multi,
            )
            save_plot(fname, kind="vqe", molecule=molecule, show=show)

        return {
            "mode": "convergence",
            "energies": results,
            "final_energies": final_vals,
        }

    # ============================================================
    # MODE 2: Noise sweep + multi-seed statistics
    # ============================================================
    if mode != "noise_stats":
        raise ValueError(f"Unknown mode '{mode}'. Use 'convergence' or 'noise_stats'.")

    if reference != "per_seed_noiseless":
        raise ValueError(
            f"Unknown reference '{reference}'. Only 'per_seed_noiseless' is supported."
        )

    if seeds is None:
        seeds = np.arange(0, 10)
    else:
        seeds = np.asarray(seeds)

    if noise_levels is None:
        noise_levels = np.arange(0.0, 0.11, 0.02)
    else:
        noise_levels = np.asarray(noise_levels)

    noise_type = str(noise_type).lower()
    _noise_probs_for_type(noise_type, 0.0)

    out = {
        "mode": "noise_stats",
        "molecule": molecule,
        "ansatz_name": ansatz_name,
        "steps": int(steps),
        "mapping": mapping,
        "noise_type": noise_type,
        "noise_levels": [float(x) for x in noise_levels],
        "seeds": [int(s) for s in seeds],
        "optimizers": {},
    }

    for opt_name in optimizers:
        lr = _stepsize_for(opt_name)
        print(f"\n⚙️ Optimizer: {opt_name} (stepsize={lr})")

        deltaE_mean, deltaE_std = [], []
        fid_mean, fid_std = [], []

        # Reference runs per seed for this optimizer (noiseless)
        print("  🔹 Computing noiseless references per seed...")
        ref_E = {}
        ref_state = {}
        for s in seeds:
            s_int = int(s)
            np.random.seed(s_int)
            ref = run_vqe(
                molecule=molecule,
                steps=steps,
                stepsize=lr,
                plot=False,
                unit=unit,
                ansatz_name=ansatz_name,
                optimizer_name=opt_name,
                noisy=False,
                mapping=mapping,
                force=force,
                seed=s_int,
            )
            ref_E[s_int] = float(ref["energy"])
            psi = np.array(ref["final_state_real"]) + 1j * np.array(
                ref["final_state_imag"]
            )
            # Normalise defensively
            psi = psi / np.linalg.norm(psi)
            ref_state[s_int] = psi

        # Noisy sweep
        print("  🔹 Sweeping noise levels...")
        for level in noise_levels:
            noise_kwargs = _noise_probs_for_type(noise_type, float(level))

            dEs = []
            Fs = []
            for s in seeds:
                s_int = int(s)
                np.random.seed(s_int)
                res = run_vqe(
                    molecule=molecule,
                    steps=steps,
                    stepsize=lr,
                    unit=unit,
                    plot=False,
                    ansatz_name=ansatz_name,
                    optimizer_name=opt_name,
                    noisy=True,
                    **noise_kwargs,
                    mapping=mapping,
                    force=force,
                    seed=s_int,
                )

                E_noisy = float(res["energy"])
                rho_or_state = np.array(res["final_state_real"]) + 1j * np.array(
                    res["final_state_imag"]
                )
                rho_or_state = (
                    rho_or_state.reshape(ref_state[s_int].shape)
                    if rho_or_state.shape == ref_state[s_int].shape
                    else rho_or_state
                )

                # Normalise statevector case (density matrix case handled in compute_fidelity)
                if rho_or_state.ndim == 1:
                    rho_or_state = rho_or_state / np.linalg.norm(rho_or_state)

                dEs.append(E_noisy - ref_E[s_int])
                Fs.append(compute_fidelity(ref_state[s_int], rho_or_state))

            dEs = np.asarray(dEs, dtype=float)
            Fs = np.asarray(Fs, dtype=float)

            deltaE_mean.append(float(np.mean(dEs)))
            deltaE_std.append(float(np.std(dEs)))
            fid_mean.append(float(np.mean(Fs)))
            fid_std.append(float(np.std(Fs)))

            print(
                f"    {_format_noise_point(noise_kwargs)}: "
                f"ΔE={deltaE_mean[-1]:.6f} ± {deltaE_std[-1]:.6f}, "
                f"⟨F⟩={fid_mean[-1]:.4f} ± {fid_std[-1]:.4f}"
            )

        out["optimizers"][opt_name] = {
            "stepsize": lr,
            "deltaE_mean": deltaE_mean,
            "deltaE_std": deltaE_std,
            "fidelity_mean": fid_mean,
            "fidelity_std": fid_std,
        }

    if plot:
        # ΔE overlay
        plt.figure(figsize=(8, 5))
        for opt_name in optimizers:
            data = out["optimizers"][opt_name]
            plt.errorbar(
                noise_levels,
                data["deltaE_mean"],
                yerr=data["deltaE_std"],
                fmt="o-",
                capsize=3,
                label=opt_name,
            )
        plt.title(f"{molecule} — ΔE vs Noise ({noise_type}, {ansatz_name})")
        plt.xlabel("Noise Probability")
        plt.ylabel("ΔE (Ha)")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()

        fname = build_filename(
            topic="noisy_optimizer_comparison_deltaE",
            ansatz=ansatz_name,
            noise_scan=True,
            noise_type=noise_type,
            multi_seed=True,
        )
        save_plot(fname, kind="vqe", molecule=molecule, show=show)

        # Fidelity overlay
        plt.figure(figsize=(8, 5))
        for opt_name in optimizers:
            data = out["optimizers"][opt_name]
            plt.errorbar(
                noise_levels,
                data["fidelity_mean"],
                yerr=data["fidelity_std"],
                fmt="s-",
                capsize=3,
                label=opt_name,
            )
        plt.title(f"{molecule} — Fidelity vs Noise ({noise_type}, {ansatz_name})")
        plt.xlabel("Noise Probability")
        plt.ylabel("Fidelity")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()

        fname = build_filename(
            topic="noisy_optimizer_comparison_fidelity",
            ansatz=ansatz_name,
            noise_scan=True,
            noise_type=noise_type,
            multi_seed=True,
        )
        save_plot(fname, kind="vqe", molecule=molecule, show=show)

    return out


# ================================================================
# ANSATZ COMPARISON
# ================================================================
def run_vqe_ansatz_comparison(
    molecule: str = "H2",
    optimizer_name: str = "Adam",
    ansatzes=None,
    steps: int = 50,
    stepsize: float | None = None,
    noisy: bool = True,
    depolarizing_prob: float = 0.05,
    amplitude_damping_prob: float = 0.05,
    phase_damping_prob: float = 0.0,
    bit_flip_prob: float = 0.0,
    phase_flip_prob: float = 0.0,
    seed: int = 0,
    mode: str = "convergence",
    noise_type: str = "depolarizing",
    noise_levels=None,
    seeds=None,
    reference: str = "per_seed_noiseless",
    force: bool = False,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    show: bool = True,
    plot: bool = True,
):
    """
    Compare ansatz families for a fixed optimizer.

    If `stepsize` is omitted, use the calibrated default for `optimizer_name`.

    mode="convergence":
        - single run per ansatz
        - returns energy trajectories vs iteration

    mode="noise_stats":
        - sweeps noise_levels and averages over seeds for each ansatz
        - computes ΔE (vs per-seed noiseless reference) and fidelity mean/std vs noise
        - sweeps one built-in channel selected by `noise_type`
    """
    import matplotlib.pyplot as plt

    from common.plotting import build_filename, save_plot

    ansatzes = ansatzes or [
        "UCCSD",
        "RY-CZ",
        "TwoQubit-RY-CNOT",
        "StronglyEntanglingLayers",
    ]
    resolved_stepsize = (
        get_optimizer_stepsize(optimizer_name) if stepsize is None else float(stepsize)
    )

    if mode == "convergence":
        results = {}
        final_vals = {}

        for ans_name in ansatzes:
            print(f"\n🔹 Running ansatz: {ans_name}")
            res = run_vqe(
                molecule=molecule,
                steps=steps,
                stepsize=resolved_stepsize,
                plot=False,
                ansatz_name=ans_name,
                optimizer_name=optimizer_name,
                noisy=noisy,
                depolarizing_prob=float(depolarizing_prob),
                amplitude_damping_prob=float(amplitude_damping_prob),
                phase_damping_prob=float(phase_damping_prob),
                bit_flip_prob=float(bit_flip_prob),
                phase_flip_prob=float(phase_flip_prob),
                mapping=mapping,
                unit=unit,
                force=force,
                seed=int(seed),
            )
            results[ans_name] = res["energies"]
            final_vals[ans_name] = res["energy"]

        if plot:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 5))
            min_len = min(len(v) for v in results.values())
            for ans, energies in results.items():
                plt.plot(range(min_len), energies[:min_len], label=ans)

            plt.title(f"{molecule} – Ansatz Comparison (opt={optimizer_name})")
            plt.xlabel("Iteration")
            plt.ylabel("Energy (Ha)")
            plt.grid(True, alpha=0.4)
            plt.legend()
            plt.tight_layout()

            fname = build_filename(
                topic="ansatz_conv",
                optimizer=optimizer_name,
                dep=depolarizing_prob if noisy else None,
                amp=amplitude_damping_prob if noisy else None,
                seed=seed,
                multi_seed=bool(seeds) and (len(seeds) > 1),
            )
            save_plot(fname, kind="vqe", molecule=molecule, show=show)

        return {
            "mode": "convergence",
            "energies": results,
            "final_energies": final_vals,
            "optimizer_name": optimizer_name,
            "steps": int(steps),
            "stepsize": resolved_stepsize,
        }

    if mode != "noise_stats":
        raise ValueError(f"Unknown mode '{mode}'. Use 'convergence' or 'noise_stats'.")

    if reference != "per_seed_noiseless":
        raise ValueError(
            f"Unknown reference '{reference}'. Only 'per_seed_noiseless' is supported."
        )

    if seeds is None:
        seeds = np.arange(0, 10)
    else:
        seeds = np.asarray(seeds)

    if noise_levels is None:
        noise_levels = np.arange(0.0, 0.11, 0.02)
    else:
        noise_levels = np.asarray(noise_levels)

    noise_type_l = str(noise_type).lower()
    _noise_probs_for_type(noise_type_l, 0.0)

    out = {
        "mode": "noise_stats",
        "molecule": molecule,
        "optimizer_name": optimizer_name,
        "steps": int(steps),
        "stepsize": resolved_stepsize,
        "mapping": mapping,
        "noise_type": noise_type_l,
        "noise_levels": [float(x) for x in noise_levels],
        "seeds": [int(s) for s in seeds],
        "ansatzes": {},
    }

    for ans_name in ansatzes:
        print(
            f"\n🔹 Ansatz: {ans_name} (optimizer={optimizer_name}, stepsize={resolved_stepsize})"
        )

        deltaE_mean, deltaE_std = [], []
        fid_mean, fid_std = [], []

        print("  🔹 Computing noiseless references per seed...")
        ref_E = {}
        ref_state = {}
        for s in seeds:
            s_int = int(s)
            np.random.seed(s_int)
            ref = run_vqe(
                molecule=molecule,
                steps=steps,
                stepsize=resolved_stepsize,
                plot=False,
                ansatz_name=ans_name,
                optimizer_name=optimizer_name,
                noisy=False,
                mapping=mapping,
                unit=unit,
                force=force,
                seed=s_int,
            )
            ref_E[s_int] = float(ref["energy"])
            psi = np.array(ref["final_state_real"]) + 1j * np.array(
                ref["final_state_imag"]
            )
            psi = psi / np.linalg.norm(psi)
            ref_state[s_int] = psi

        print("  🔹 Sweeping noise levels...")
        for level in noise_levels:
            noise_kwargs = _noise_probs_for_type(noise_type_l, float(level))

            dEs = []
            Fs = []
            for s in seeds:
                s_int = int(s)
                np.random.seed(s_int)
                res = run_vqe(
                    molecule=molecule,
                    steps=steps,
                    stepsize=resolved_stepsize,
                    plot=False,
                    ansatz_name=ans_name,
                    optimizer_name=optimizer_name,
                    noisy=True,
                    **noise_kwargs,
                    mapping=mapping,
                    unit=unit,
                    force=force,
                    seed=s_int,
                )

                E_noisy = float(res["energy"])
                state_or_rho = np.array(res["final_state_real"]) + 1j * np.array(
                    res["final_state_imag"]
                )

                if state_or_rho.ndim == 1:
                    state_or_rho = state_or_rho / np.linalg.norm(state_or_rho)

                dEs.append(E_noisy - ref_E[s_int])
                Fs.append(compute_fidelity(ref_state[s_int], state_or_rho))

            dEs = np.asarray(dEs, dtype=float)
            Fs = np.asarray(Fs, dtype=float)

            deltaE_mean.append(float(np.mean(dEs)))
            deltaE_std.append(float(np.std(dEs)))
            fid_mean.append(float(np.mean(Fs)))
            fid_std.append(float(np.std(Fs)))

            print(
                f"    {_format_noise_point(noise_kwargs)}: "
                f"ΔE={deltaE_mean[-1]:.6f} ± {deltaE_std[-1]:.6f}, "
                f"⟨F⟩={fid_mean[-1]:.4f} ± {fid_std[-1]:.4f}"
            )

        out["ansatzes"][ans_name] = {
            "deltaE_mean": deltaE_mean,
            "deltaE_std": deltaE_std,
            "fidelity_mean": fid_mean,
            "fidelity_std": fid_std,
        }

    if plot:
        plt.figure(figsize=(8, 5))
        for ans_name in ansatzes:
            data = out["ansatzes"][ans_name]
            plt.errorbar(
                noise_levels,
                data["deltaE_mean"],
                yerr=data["deltaE_std"],
                fmt="o-",
                capsize=3,
                label=ans_name,
            )
        plt.title(f"{molecule} — ΔE vs Noise ({noise_type_l}, opt={optimizer_name})")
        plt.xlabel("Noise Probability")
        plt.ylabel("ΔE (Ha)")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()

        fname = build_filename(
            topic="noisy_ansatz_comparison_deltaE",
            optimizer=optimizer_name,
            noise_scan=True,
            noise_type=noise_type_l,
            multi_seed=True,
        )
        save_plot(fname, kind="vqe", molecule=molecule, show=show)

        plt.figure(figsize=(8, 5))
        for ans_name in ansatzes:
            data = out["ansatzes"][ans_name]
            plt.errorbar(
                noise_levels,
                data["fidelity_mean"],
                yerr=data["fidelity_std"],
                fmt="s-",
                capsize=3,
                label=ans_name,
            )
        plt.title(
            f"{molecule} — Fidelity vs Noise ({noise_type_l}, opt={optimizer_name})"
        )
        plt.xlabel("Noise Probability")
        plt.ylabel("Fidelity")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()

        fname = build_filename(
            topic="noisy_ansatz_comparison_fidelity",
            optimizer=optimizer_name,
            noise_scan=True,
            noise_type=noise_type_l,
            multi_seed=True,
        )
        save_plot(fname, kind="vqe", molecule=molecule, show=show)

    return out


# ================================================================
# MULTI-SEED NOISE STUDIES
# ================================================================
def run_vqe_multi_seed_noise(
    molecule="H2",
    ansatz_name="RY-CZ",
    optimizer_name="Adam",
    steps=30,
    stepsize=None,
    seeds=None,
    noise_type="depolarizing",
    depolarizing_probs=None,
    amplitude_damping_probs=None,
    force=False,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    show: bool = True,
):
    """
    Multi-seed noise statistics for a given molecule and ansatz.
    """
    if seeds is None:
        seeds = np.arange(0, 5)

    if depolarizing_probs is None:
        depolarizing_probs = np.arange(0.0, 0.11, 0.02)

    if amplitude_damping_probs is None:
        amplitude_damping_probs = np.zeros_like(depolarizing_probs)
    resolved_stepsize = (
        get_optimizer_stepsize(optimizer_name) if stepsize is None else float(stepsize)
    )
    noise_type_norm = str(noise_type).strip().lower()
    _noise_probs_for_type(noise_type_norm, 0.0)

    print("\n🔹 Computing noiseless reference runs...")
    ref_energies, ref_states = [], []
    for s in seeds:
        np.random.seed(int(s))
        res = run_vqe(
            molecule=molecule,
            steps=steps,
            stepsize=resolved_stepsize,
            plot=False,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            noisy=False,
            mapping=mapping,
            unit=unit,
            force=force,
            seed=int(s),
        )
        ref_energies.append(res["energy"])
        state = np.array(res["final_state_real"]) + 1j * np.array(
            res["final_state_imag"]
        )
        ref_states.append(state)

    reference_energy = float(np.mean(ref_energies))
    reference_state = ref_states[0] / np.linalg.norm(ref_states[0])
    print(f"Reference mean energy = {reference_energy:.6f} Ha")

    # --- Noisy sweeps ---
    energy_means, energy_stds = [], []
    fidelity_means, fidelity_stds = [], []

    for level in depolarizing_probs:
        noise_kwargs = _noise_probs_for_type(noise_type_norm, float(level))
        noisy_energies, fidelities = [], []
        for s in seeds:
            np.random.seed(int(s))
            res = run_vqe(
                molecule=molecule,
                steps=steps,
                stepsize=resolved_stepsize,
                plot=False,
                ansatz_name=ansatz_name,
                optimizer_name=optimizer_name,
                noisy=True,
                **noise_kwargs,
                mapping=mapping,
                unit=unit,
                force=force,
                seed=int(s),
            )
            noisy_energies.append(res["energy"])
            state = np.array(res["final_state_real"]) + 1j * np.array(
                res["final_state_imag"]
            )
            state = state / np.linalg.norm(state)
            fidelities.append(compute_fidelity(reference_state, state))

        noisy_energies = np.array(noisy_energies)
        dE = noisy_energies - reference_energy

        energy_means.append(float(np.mean(dE)))
        energy_stds.append(float(np.std(dE)))
        fidelity_means.append(float(np.mean(fidelities)))
        fidelity_stds.append(float(np.std(fidelities)))

        print(
            f"Noise {_format_noise_point(noise_kwargs)}: "
            f"ΔE={energy_means[-1]:.6f} ± {energy_stds[-1]:.6f}, "
            f"⟨F⟩={fidelity_means[-1]:.4f}"
        )

    noise_levels = depolarizing_probs

    plot_noise_statistics(
        molecule,
        noise_levels,
        energy_means,
        energy_stds,
        fidelity_means,
        fidelity_stds,
        optimizer_name=optimizer_name,
        ansatz_name=ansatz_name,
        noise_type=noise_type_norm,
        show=show,
    )

    print(f"\n✅ Multi-seed noise study complete for {molecule}")


# ================================================================
# GEOMETRY SCAN
# ================================================================
def run_vqe_geometry_scan(
    molecule="H2_BOND",
    param_name="bond",
    param_values=None,
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    steps=30,
    stepsize=None,
    seeds=None,
    force=False,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    show: bool = True,
):
    """
    Geometry scan using run_vqe + generate_geometry, mirroring the H₂O and LiH notebooks.

    Parameters
    ----------
    show : bool
        Whether to display the generated plot.

    Returns
    -------
    list of tuples
        [(param_value, mean_E, std_E), ...]
    """
    import matplotlib.pyplot as plt

    from common.plotting import (
        build_filename,
        save_plot,
    )

    if param_values is None:
        raise ValueError("param_values must be specified")

    seeds = seeds or [0]
    resolved_stepsize = (
        get_optimizer_stepsize(optimizer_name) if stepsize is None else float(stepsize)
    )
    results = []

    for val in param_values:
        print(f"\n⚙️ Geometry: {param_name} = {val:.3f}")
        symbols, coordinates = generate_geometry(molecule, val, unit=unit)

        energies_for_val = []
        for s in seeds:
            np.random.seed(int(s))
            res = run_vqe(
                molecule=molecule,
                steps=steps,
                stepsize=resolved_stepsize,
                ansatz_name=ansatz_name,
                optimizer_name=optimizer_name,
                symbols=symbols,
                coordinates=coordinates,
                noisy=False,
                plot=False,
                seed=int(s),
                force=force,
                mapping=mapping,
                unit=unit,
                active_electrons=active_electrons,
                active_orbitals=active_orbitals,
            )
            energies_for_val.append(res["energy"])

        mean_E = float(np.mean(energies_for_val))
        std_E = float(np.std(energies_for_val))
        results.append((val, mean_E, std_E))
        print(f"  → Mean E = {mean_E:.6f} ± {std_E:.6f} Ha")

    # --- Plot ---
    params, means, stds = zip(*results)

    plt.errorbar(params, means, yerr=stds, fmt="o-", capsize=4)
    if str(param_name).strip().lower() == "angle":
        xlabel = f"{param_name.capitalize()} (deg)"
    else:
        xlabel = f"{param_name.capitalize()} ({coordinate_unit_label(unit)})"
    plt.xlabel(xlabel)
    plt.ylabel("Ground-State Energy (Ha)")
    plt.title(
        f"{molecule} Energy vs {param_name.capitalize()} ({ansatz_name}, {optimizer_name})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = build_filename(
        topic=f"vqe_geometry_scan_{param_name}",
        ansatz=ansatz_name,
        optimizer=optimizer_name,
        multi_seed=True,
    )
    save_plot(fname, kind="vqe", molecule=molecule, show=show)

    min_idx = int(np.argmin(means))
    print(
        f"Minimum energy: {means[min_idx]:.6f} ± {stds[min_idx]:.6f} "
        f"at {param_name}={params[min_idx]:.3f}"
    )

    return results


# ================================================================
# MAPPING COMPARISON
# ================================================================
def run_vqe_mapping_comparison(
    molecule="H2",
    ansatz_name="UCCSD",
    optimizer_name="Adam",
    mappings=None,
    steps=50,
    stepsize=None,
    noisy=False,
    depolarizing_prob=0.0,
    amplitude_damping_prob=0.0,
    force=False,
    show=True,
    seed=0,
    unit: str = "angstrom",
):
    """
    Compare different fermion-to-qubit mappings by:

    - Building qubit Hamiltonians via build_hamiltonian
    - Running VQE (re-using caching) via run_vqe for each mapping
    - Plotting energy convergence curves and printing summary

    Parameters
    ----------
    show : bool
        Whether to display the generated plot.

    Returns
    -------
    dict
        {
            mapping_name: {
                "final_energy": float,
                "energies": [...],
                "num_qubits": int,
                "num_terms": int or None,
            },
            ...
        }
    """
    import matplotlib.pyplot as plt

    from common.plotting import build_filename, save_plot

    np.random.seed(seed)
    resolved_stepsize = (
        get_optimizer_stepsize(optimizer_name) if stepsize is None else float(stepsize)
    )

    mappings = mappings or ["jordan_wigner", "bravyi_kitaev", "parity"]
    results = {}

    print(f"\n🔍 Comparing mappings for {molecule} ({ansatz_name}, {optimizer_name})")

    for mapping in mappings:
        print(f"\n⚙️ Running mapping: {mapping}")

        # Build Hamiltonian once to inspect complexity
        H, qubits, hf_state, symbols, coordinates, basis, charge, unit_out = (
            build_hamiltonian(molecule, mapping=mapping, unit=unit)
        )
        basis = basis.lower()

        try:
            num_terms = len(H.ops)
        except AttributeError:
            try:
                num_terms = len(H.terms()[0]) if callable(H.terms) else len(H.data)
            except Exception:
                num_terms = len(getattr(H, "data", [])) if hasattr(H, "data") else None

        # Run VQE using the high-level entrypoint (handles ansatz + noise plumbing)
        res = run_vqe(
            molecule=molecule,
            ansatz_name=ansatz_name,
            optimizer_name=optimizer_name,
            steps=steps,
            stepsize=resolved_stepsize,
            unit=unit,
            noisy=noisy,
            depolarizing_prob=depolarizing_prob,
            amplitude_damping_prob=amplitude_damping_prob,
            mapping=mapping,
            force=force,
            plot=False,
            seed=seed,
        )

        results[mapping] = {
            "final_energy": res["energy"],
            "energies": res["energies"],
            "num_qubits": qubits,
            "num_terms": num_terms,
        }

    # --- Plot mappings ---
    plt.figure(figsize=(8, 5))
    for mapping in mappings:
        data = results[mapping]
        label = mapping.replace("_", "-").title()
        plt.plot(
            range(len(data["energies"])),
            data["energies"],
            label=label,
            linewidth=2,
            alpha=0.9,
        )

    plt.xlabel("Iteration")
    plt.ylabel("Energy (Ha)")
    plt.title(f"{molecule} VQE: Energy Convergence by Mapping ({ansatz_name})")
    plt.legend(frameon=False, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(pad=2)

    fname = build_filename(
        topic="mapping_comparison",
        ansatz=ansatz_name,
        optimizer=optimizer_name,
        multi_seed=True,
    )
    path = save_plot(fname, kind="vqe", molecule=molecule, show=show)

    print(f"\n📉 Saved mapping comparison plot → {path}\nResults Summary:")

    for mapping, data in results.items():
        print(
            f"  {mapping:15s} → E = {data['final_energy']:.8f} Ha, "
            f"Qubits = {data['num_qubits']}, Terms = {data['num_terms']}"
        )

    return results
