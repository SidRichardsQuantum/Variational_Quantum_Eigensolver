"""
qite.core
=========
High-level orchestration for VarQITE (McLachlan variational imaginary-time evolution).

This module mirrors the ergonomics of vqe.core and qpe.core:

- Cached main entrypoint:          run_qite(...)
- Optional plotting + saving:      qite.visualize
- Reproducible I/O + hashing:      qite.io_utils
- Circuit plumbing / QNodes:       qite.engine

Important
---------
VarQITE requires a pure statevector, so noisy/mixed-state runs are not supported.
Noise is supported only in the CLI's post-evaluation mode (see qite.__main__).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import pennylane as qml
from pennylane import numpy as np

from common.persist import cached_compute_runtime, canonical_hamiltonian
from common.problem import problem_metadata, resolve_problem
from common.termination import optimize, stopping_config, supplied_parameters
from qite.engine import build_ansatz as engine_build_ansatz
from qite.engine import (
    make_device,
    make_energy_qnode,
    make_state_qnode,
    qite_step,
    qrte_step,
    validate_solver,
)
from qite.io_utils import (
    ensure_dirs,
    load_run_record,
    make_filename_prefix,
    make_run_config_dict,
    run_signature,
    save_run_record,
)
from qite.visualize import plot_convergence
from vqe.auto_ansatz import resolve_auto_ansatz


def run_qite(
    molecule: str = "H2",
    *,
    seed: int = 0,
    steps: int = 75,
    dtau: float = 0.2,
    plot: bool = True,
    ansatz_name: str = "UCCSD",
    ansatz_kwargs: dict[str, Any] | None = None,
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    show: bool = True,
    fd_eps: float = 1e-3,
    reg: float = 1e-6,
    solver: str = "solve",
    pinv_rcond: float = 1e-10,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    phase_damping_prob: float = 0.0,
    bit_flip_prob: float = 0.0,
    phase_flip_prob: float = 0.0,
    noise_model=None,
    hamiltonian: qml.Hamiltonian | None = None,
    num_qubits: int | None = None,
    reference_state=None,
    energy_tol: float | None = None,
    patience: int = 1,
    initial_params=None,
    initialization_source: dict | None = None,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run VarQITE end-to-end with caching.

    VarQITE uses a McLachlan linear-system update requiring a pure statevector.
    Noisy/mixed-state runs are intentionally not supported here.

    Returns
    -------
    dict
        {
            "energy": float,
            "energies": [float, ...],
            "steps": int,
            "dtau": float,
            "num_qubits": int,
            "final_state_real": [...],
            "final_state_imag": [...],
            "final_params": [...],
            "final_params_shape": [...],
            "varqite": {...},
        }
    """
    stopping = stopping_config(steps, energy_tol, patience)
    if initial_params is not None and not np.all(
        np.isfinite(np.asarray(initial_params, dtype=float))
    ):
        raise ValueError("initial_params must be finite")
    if initialization_source is not None and initial_params is None:
        raise ValueError("initialization_source requires initial_params")
    start_time = time.perf_counter()
    solver = validate_solver(solver)
    ensure_dirs()
    np.random.seed(int(seed))

    if (
        bool(noisy)
        or (float(depolarizing_prob) != 0.0)
        or (float(amplitude_damping_prob) != 0.0)
        or (float(phase_damping_prob) != 0.0)
        or (float(bit_flip_prob) != 0.0)
        or (float(phase_flip_prob) != 0.0)
        or (noise_model is not None)
    ):
        raise ValueError(
            "VarQITE requires a pure statevector and is not supported with "
            "noisy/mixed-state simulation. Use the CLI's eval-noise mode for "
            "post-evaluation under noise."
        )

    problem = resolve_problem(
        molecule=molecule,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        mapping=mapping,
        unit=unit,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian=hamiltonian,
        num_qubits=num_qubits,
        reference_state=reference_state,
        default_reference_state=True,
        reference_name="reference_state",
    )
    H = problem.hamiltonian
    qubits = problem.num_qubits
    hf_state = np.array(problem.reference_state, dtype=int)
    symbols_out = problem.symbols
    coordinates_out = problem.coordinates
    basis_out = problem.basis
    charge_out = problem.charge
    mapping_out = problem.mapping
    unit_out = problem.unit
    molecule_label = problem.molecule_label
    resolved_active_electrons = problem.active_electrons
    resolved_active_orbitals = problem.active_orbitals
    hamiltonian_mode = hamiltonian is not None
    cache_enabled = bool(problem.cacheable or hamiltonian_mode)
    resolved_ansatz_name, resolved_ansatz_kwargs, ansatz_selection = (
        resolve_auto_ansatz(
            str(ansatz_name),
            H,
            int(qubits),
            ansatz_kwargs=ansatz_kwargs,
        )
    )

    # --- Configuration & caching ---
    cfg = make_run_config_dict(
        symbols=symbols_out,
        coordinates=np.array(coordinates_out, dtype=float),
        basis=str(basis_out),
        charge=int(charge_out),
        unit=str(unit_out),
        seed=int(seed),
        mapping=str(mapping_out),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        dtau=float(dtau),
        steps=int(steps),
        molecule_label=molecule_label,
        ansatz_name=str(resolved_ansatz_name),
        noise_model_name=None,
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        fd_eps=float(fd_eps),
        reg=float(reg),
        solver=str(solver),
        pinv_rcond=float(pinv_rcond),
        ansatz_kwargs=resolved_ansatz_kwargs,
    )
    cfg["multiplicity"] = problem.multiplicity
    cfg.update(problem_metadata(problem))
    cfg["initialization_source"] = initialization_source
    cfg["termination_schema"] = 1
    cfg["stopping"] = stopping
    cfg["initial_params"] = (
        None
        if initial_params is None
        else np.asarray(initial_params, dtype=float).tolist()
    )
    if ansatz_selection is not None:
        cfg["ansatz_selection"] = dict(ansatz_selection)
    if hamiltonian_mode:
        cfg["hamiltonian"] = canonical_hamiltonian(H)
        cfg["num_qubits"] = int(qubits)
        cfg["reference_state"] = np.array(hf_state, dtype=int).tolist()

    prefix = None
    if cache_enabled:
        sig = run_signature(cfg)
        prefix = make_filename_prefix(
            cfg, noisy=False, seed=int(seed), hash_str=sig, algo="varqite"
        )

        if not force:
            record = load_run_record(prefix)
            res: Dict[str, Any] | None = None
            if record is not None:
                res = dict(record["result"])
                if "final_params" not in res or "final_params_shape" not in res:
                    raise KeyError(
                        "Cached VarQITE record is missing final parameters. "
                        "Re-run with force=True to refresh the cache."
                    )
                cached_compute = cached_compute_runtime(res)
                if cached_compute is None:
                    res = None
                else:
                    res["compute_runtime_s"] = cached_compute
            if record is not None and res is not None:
                res["runtime_s"] = float(time.perf_counter() - start_time)
                res["cache_hit"] = True
                if progress_callback is not None:
                    progress_callback(
                        {
                            "phase": "cache_hit",
                            "iteration": res["steps"],
                            "total_iterations": int(steps),
                            "energy": res["energy"],
                        }
                    )
                return res

    # --- Device, ansatz, QNodes ---
    dev = make_device(int(qubits), noisy=False)

    ansatz_fn, params = engine_build_ansatz(
        str(resolved_ansatz_name),
        int(qubits),
        mapping=mapping_out,
        multiplicity=problem.multiplicity,
        seed=int(seed),
        symbols=symbols_out,
        coordinates=np.array(coordinates_out, dtype=float),
        charge=int(charge_out),
        basis=str(basis_out),
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        requires_grad=True,
        hf_state=np.array(hf_state, dtype=int),
        ansatz_kwargs=resolved_ansatz_kwargs,
    )

    energy_qnode = make_energy_qnode(
        H,
        dev,
        ansatz_fn,
        int(qubits),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        noise_model=None,
    )

    state_qnode = make_state_qnode(
        dev,
        ansatz_fn,
        int(qubits),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        noise_model=None,
    )

    # --- Iteration loop (VarQITE) ---
    if initial_params is not None:
        params = supplied_parameters(initial_params, params)
    params = np.array(params, requires_grad=True)
    engine_cache: dict[str, Any] = {}

    def update(current):
        return qite_step(
            params=current,
            energy_qnode=energy_qnode,
            state_qnode=state_qnode,
            dtau=float(dtau),
            num_wires=int(qubits),
            hamiltonian=H,
            fd_eps=float(fd_eps),
            reg=float(reg),
            solver=str(solver),
            pinv_rcond=float(pinv_rcond),
            cache=engine_cache,
        )

    params, energies, params_history, termination = optimize(
        params,
        energy_qnode,
        update,
        steps=steps,
        energy_tol=energy_tol,
        patience=patience,
        progress_callback=progress_callback,
    )
    final_energy = energies[-1] if energies else None
    final_state = state_qnode(params) if energies else None
    if final_state is not None and not np.all(np.isfinite(final_state)):
        termination.update(reason="numerical_failure", message="Non-finite final state")
        final_state = None

    # --- Optional plot ---
    if plot and energies:
        plot_convergence(
            energies,
            molecule=str(molecule_label),
            method="VarQITE",
            ansatz=str(resolved_ansatz_name),
            seed=int(seed),
            dep_prob=0.0,
            amp_prob=0.0,
            noise_type=None,
            show=bool(show),
            save=True,
        )

    # --- Save ---
    params_arr = np.array(params)
    compute_runtime_s = float(time.perf_counter() - start_time)
    result = {
        "molecule": str(molecule_label),
        "mapping": str(mapping_out),
        "unit": str(unit_out),
        "charge": int(charge_out),
        "multiplicity": problem.multiplicity,
        "basis": str(basis_out),
        "active_electrons": resolved_active_electrons,
        "active_orbitals": resolved_active_orbitals,
        "ansatz": str(resolved_ansatz_name),
        "ansatz_kwargs": dict(cfg.get("ansatz_kwargs", {})),
        "energy": final_energy,
        "energies": [float(e) for e in energies],
        "steps": termination["updates"],
        "termination": termination,
        "config": cfg,
        "params_history": params_history,
        "initialization": {
            "source": "supplied" if initial_params is not None else "seed",
            "seed": int(seed),
            "provenance": initialization_source,
        },
        "dtau": float(dtau),
        "num_qubits": int(qubits),
        "final_state_real": (
            None if final_state is None else np.real(final_state).tolist()
        ),
        "final_state_imag": (
            None if final_state is None else np.imag(final_state).tolist()
        ),
        "final_params": params_arr.astype(float).ravel().tolist(),
        "final_params_shape": list(params_arr.shape),
        "varqite": {
            "fd_eps": float(fd_eps),
            "reg": float(reg),
            "solver": str(solver),
            "solver_history": engine_cache.get("solver_history", []),
            "pinv_rcond": float(pinv_rcond),
        },
        "runtime_s": compute_runtime_s,
        "compute_runtime_s": compute_runtime_s,
        "cache_hit": False,
    }
    if ansatz_selection is not None:
        result["ansatz_selection"] = dict(ansatz_selection)

    record = {"config": cfg, "result": result}
    if cache_enabled and prefix is not None:
        save_run_record(prefix, record)
        print(f"\n💾 Saved run record: {prefix}.json\n")

    return result


def run_qrte(
    molecule: str = "H2",
    *,
    seed: int = 0,
    steps: int = 50,
    dt: float = 0.05,
    plot: bool = True,
    ansatz_name: str = "UCCSD",
    ansatz_kwargs: dict[str, Any] | None = None,
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
    show: bool = True,
    fd_eps: float = 1e-3,
    reg: float = 1e-6,
    solver: str = "solve",
    pinv_rcond: float = 1e-10,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    phase_damping_prob: float = 0.0,
    bit_flip_prob: float = 0.0,
    phase_flip_prob: float = 0.0,
    noise_model=None,
    initial_params=None,
    hamiltonian: qml.Hamiltonian | None = None,
    num_qubits: int | None = None,
    reference_state=None,
) -> Dict[str, Any]:
    """
    Run VarQRTE end-to-end with caching.

    VarQRTE uses the real-time McLachlan projected update on a pure-state ansatz.
    Noisy/mixed-state optimization is intentionally not supported here.
    """
    start_time = time.perf_counter()
    solver = validate_solver(solver)
    ensure_dirs()
    np.random.seed(int(seed))

    if (
        bool(noisy)
        or (float(depolarizing_prob) != 0.0)
        or (float(amplitude_damping_prob) != 0.0)
        or (float(phase_damping_prob) != 0.0)
        or (float(bit_flip_prob) != 0.0)
        or (float(phase_flip_prob) != 0.0)
        or (noise_model is not None)
    ):
        raise ValueError(
            "VarQRTE requires a pure statevector and is not supported with "
            "noisy/mixed-state simulation."
        )

    problem = resolve_problem(
        molecule=molecule,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
        mapping=mapping,
        unit=unit,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
        hamiltonian=hamiltonian,
        num_qubits=num_qubits,
        reference_state=reference_state,
        default_reference_state=True,
        reference_name="reference_state",
    )
    H = problem.hamiltonian
    qubits = problem.num_qubits
    hf_state = np.array(problem.reference_state, dtype=int)
    symbols_out = problem.symbols
    coordinates_out = problem.coordinates
    basis_out = problem.basis
    charge_out = problem.charge
    mapping_out = problem.mapping
    unit_out = problem.unit
    molecule_label = problem.molecule_label
    resolved_active_electrons = problem.active_electrons
    resolved_active_orbitals = problem.active_orbitals
    hamiltonian_mode = hamiltonian is not None
    cache_enabled = bool(problem.cacheable or hamiltonian_mode)
    resolved_ansatz_name, resolved_ansatz_kwargs, ansatz_selection = (
        resolve_auto_ansatz(
            str(ansatz_name),
            H,
            int(qubits),
            ansatz_kwargs=ansatz_kwargs,
        )
    )

    dev = make_device(int(qubits), noisy=False)

    ansatz_fn, params = engine_build_ansatz(
        str(resolved_ansatz_name),
        int(qubits),
        mapping=mapping_out,
        multiplicity=problem.multiplicity,
        seed=int(seed),
        symbols=symbols_out,
        coordinates=np.array(coordinates_out, dtype=float),
        charge=int(charge_out),
        basis=str(basis_out),
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        requires_grad=True,
        hf_state=np.array(hf_state, dtype=int),
        ansatz_kwargs=resolved_ansatz_kwargs,
    )

    init_mode = "default"
    if initial_params is not None:
        params0 = np.array(params, dtype=float)
        provided = np.array(initial_params, dtype=float)
        if provided.size != params0.size:
            raise ValueError(
                "initial_params has the wrong size for the selected ansatz: "
                f"expected {params0.size}, got {provided.size}."
            )
        params = np.array(
            provided.reshape(params0.shape),
            requires_grad=True,
        )
        init_mode = "provided"

    cfg = make_run_config_dict(
        symbols=symbols_out,
        coordinates=np.array(coordinates_out, dtype=float),
        basis=str(basis_out),
        charge=int(charge_out),
        unit=str(unit_out),
        seed=int(seed),
        mapping=str(mapping_out),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        dtau=float(dt),
        steps=int(steps),
        molecule_label=molecule_label,
        ansatz_name=str(resolved_ansatz_name),
        noise_model_name=None,
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        fd_eps=float(fd_eps),
        reg=float(reg),
        solver=str(solver),
        pinv_rcond=float(pinv_rcond),
        ansatz_kwargs=resolved_ansatz_kwargs,
    )
    cfg["time_mode"] = "real"
    cfg["initialization"] = init_mode
    cfg["multiplicity"] = problem.multiplicity
    if ansatz_selection is not None:
        cfg["ansatz_selection"] = dict(ansatz_selection)
    if hamiltonian_mode:
        cfg["hamiltonian"] = canonical_hamiltonian(H)
        cfg["num_qubits"] = int(qubits)
        cfg["reference_state"] = np.array(hf_state, dtype=int).tolist()
    if initial_params is not None:
        cfg["initial_params"] = np.round(
            np.array(initial_params, dtype=float).ravel(),
            8,
        ).tolist()

    prefix = None
    if cache_enabled:
        sig = run_signature(cfg)
        prefix = make_filename_prefix(
            cfg, noisy=False, seed=int(seed), hash_str=sig, algo="varqrte"
        )

        if not force:
            record = load_run_record(prefix)
            res: Dict[str, Any] | None = None
            if record is not None:
                res = dict(record["result"])
                if "final_params" not in res or "final_params_shape" not in res:
                    raise KeyError(
                        "Cached VarQRTE record is missing final parameters. "
                        "Re-run with force=True to refresh the cache."
                    )
                cached_compute = cached_compute_runtime(res)
                if cached_compute is None:
                    res = None
                else:
                    res["compute_runtime_s"] = cached_compute
            if record is not None and res is not None:
                res["runtime_s"] = float(time.perf_counter() - start_time)
                res["cache_hit"] = True
                return res

    energy_qnode = make_energy_qnode(
        H,
        dev,
        ansatz_fn,
        int(qubits),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        noise_model=None,
    )

    state_qnode = make_state_qnode(
        dev,
        ansatz_fn,
        int(qubits),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        noise_model=None,
    )

    params = np.array(params, requires_grad=True)
    energies = [float(energy_qnode(params))]
    times = [0.0]
    params_history: list[list[float]] = [np.array(params, dtype=float).ravel().tolist()]

    engine_cache: dict[str, Any] = {}
    print("\n⚙️ Using VarQRTE (McLachlan real-time) update rule")

    for k in range(int(steps)):
        params = qrte_step(
            params=params,
            energy_qnode=energy_qnode,
            state_qnode=state_qnode,
            dt=float(dt),
            num_wires=int(qubits),
            hamiltonian=H,
            fd_eps=float(fd_eps),
            reg=float(reg),
            solver=str(solver),
            pinv_rcond=float(pinv_rcond),
            cache=engine_cache,
        )

        e = float(energy_qnode(params))
        energies.append(e)
        times.append(float((k + 1) * float(dt)))
        params_history.append(np.array(params, dtype=float).ravel().tolist())
        print(f"Iter {k + 1:02d}/{steps}: E = {e:.6f} Ha")

    final_energy = float(energies[-1])
    final_state = state_qnode(params)

    if plot:
        plot_convergence(
            energies,
            molecule=str(molecule_label),
            method="VarQRTE",
            ansatz=str(resolved_ansatz_name),
            step_label="Time Step",
            seed=int(seed),
            dep_prob=0.0,
            amp_prob=0.0,
            noise_type=None,
            show=bool(show),
            save=True,
        )

    params_arr = np.array(params)
    compute_runtime_s = float(time.perf_counter() - start_time)
    result = {
        "molecule": str(molecule_label),
        "mapping": str(mapping_out),
        "unit": str(unit_out),
        "charge": int(charge_out),
        "multiplicity": problem.multiplicity,
        "basis": str(basis_out),
        "active_electrons": resolved_active_electrons,
        "active_orbitals": resolved_active_orbitals,
        "ansatz": str(resolved_ansatz_name),
        "ansatz_kwargs": dict(cfg.get("ansatz_kwargs", {})),
        "energy": float(final_energy),
        "energies": [float(e) for e in energies],
        "times": [float(t) for t in times],
        "steps": int(steps),
        "dt": float(dt),
        "num_qubits": int(qubits),
        "final_state_real": np.real(final_state).tolist(),
        "final_state_imag": np.imag(final_state).tolist(),
        "final_params": params_arr.astype(float).ravel().tolist(),
        "final_params_shape": list(params_arr.shape),
        "params_history": params_history,
        "initialization": init_mode,
        "varqrte": {
            "fd_eps": float(fd_eps),
            "reg": float(reg),
            "solver": str(solver),
            "solver_history": engine_cache.get("solver_history", []),
            "pinv_rcond": float(pinv_rcond),
        },
        "runtime_s": compute_runtime_s,
        "compute_runtime_s": compute_runtime_s,
        "cache_hit": False,
    }
    if ansatz_selection is not None:
        result["ansatz_selection"] = dict(ansatz_selection)

    record = {"config": cfg, "result": result}
    if cache_enabled and prefix is not None:
        save_run_record(prefix, record)
        print(f"\n💾 Saved run record: {prefix}.json\n")

    return result
