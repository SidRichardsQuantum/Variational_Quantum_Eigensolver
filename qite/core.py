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

from typing import Any, Dict

import pennylane as qml
from pennylane import numpy as np

from common.problem import resolve_problem
from qite.engine import build_ansatz as engine_build_ansatz
from qite.engine import (
    make_device,
    make_energy_qnode,
    make_state_qnode,
    qite_step,
    qrte_step,
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


def compute_fidelity(pure_state, state_or_rho) -> float:
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

    if state_or_rho.ndim == 2:
        return float(np.real(np.vdot(pure_state, state_or_rho @ pure_state)))

    raise ValueError("Invalid state shape for fidelity computation")


def run_qite(
    molecule: str = "H2",
    *,
    seed: int = 0,
    steps: int = 50,
    dtau: float = 0.2,
    plot: bool = True,
    ansatz_name: str = "UCCSD",
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    charge: int = 0,
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
    cache_enabled = problem.cacheable

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
        ansatz_name=str(ansatz_name),
        noise_model_name=None,
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        fd_eps=float(fd_eps),
        reg=float(reg),
        solver=str(solver),
        pinv_rcond=float(pinv_rcond),
    )

    prefix = None
    if cache_enabled:
        sig = run_signature(cfg)
        prefix = make_filename_prefix(
            cfg, noisy=False, seed=int(seed), hash_str=sig, algo="varqite"
        )

        if not force:
            record = load_run_record(prefix)
            if record is not None:
                res = record["result"]
                if "final_params" not in res or "final_params_shape" not in res:
                    raise KeyError(
                        "Cached VarQITE record is missing final parameters. "
                        "Re-run with force=True to refresh the cache."
                    )
                return res

    # --- Device, ansatz, QNodes ---
    dev = make_device(int(qubits), noisy=False)

    ansatz_fn, params = engine_build_ansatz(
        str(ansatz_name),
        int(qubits),
        seed=int(seed),
        symbols=symbols_out,
        coordinates=np.array(coordinates_out, dtype=float),
        charge=int(charge_out),
        basis=str(basis_out),
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        requires_grad=True,
        hf_state=np.array(hf_state, dtype=int),
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
    params = np.array(params, requires_grad=True)
    energies = [float(energy_qnode(params))]

    engine_cache: dict[str, Any] = {}
    print("\n⚙️ Using VarQITE (McLachlan) update rule")

    for k in range(int(steps)):
        params = qite_step(
            params=params,
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

        e = float(energy_qnode(params))
        energies.append(e)
        print(f"Iter {k + 1:02d}/{steps}: E = {e:.6f} Ha")

    final_energy = float(energies[-1])
    final_state = state_qnode(params)

    # --- Optional plot ---
    if plot:
        plot_convergence(
            energies,
            molecule=str(molecule_label),
            method="VarQITE",
            ansatz=str(ansatz_name),
            seed=int(seed),
            dep_prob=0.0,
            amp_prob=0.0,
            noise_type=None,
            show=bool(show),
            save=True,
        )

    # --- Save ---
    params_arr = np.array(params)
    result = {
        "molecule": str(molecule_label),
        "mapping": str(mapping_out),
        "unit": str(unit_out),
        "charge": int(charge_out),
        "basis": str(basis_out),
        "active_electrons": resolved_active_electrons,
        "active_orbitals": resolved_active_orbitals,
        "ansatz": str(ansatz_name),
        "energy": float(final_energy),
        "energies": [float(e) for e in energies],
        "steps": int(steps),
        "dtau": float(dtau),
        "num_qubits": int(qubits),
        "final_state_real": np.real(final_state).tolist(),
        "final_state_imag": np.imag(final_state).tolist(),
        "final_params": params_arr.astype(float).ravel().tolist(),
        "final_params_shape": list(params_arr.shape),
        "varqite": {
            "fd_eps": float(fd_eps),
            "reg": float(reg),
            "solver": str(solver),
            "pinv_rcond": float(pinv_rcond),
        },
    }

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
    force: bool = False,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    charge: int = 0,
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
    cache_enabled = problem.cacheable

    dev = make_device(int(qubits), noisy=False)

    ansatz_fn, params = engine_build_ansatz(
        str(ansatz_name),
        int(qubits),
        seed=int(seed),
        symbols=symbols_out,
        coordinates=np.array(coordinates_out, dtype=float),
        charge=int(charge_out),
        basis=str(basis_out),
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        requires_grad=True,
        hf_state=np.array(hf_state, dtype=int),
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
        ansatz_name=str(ansatz_name),
        noise_model_name=None,
        active_electrons=resolved_active_electrons,
        active_orbitals=resolved_active_orbitals,
        fd_eps=float(fd_eps),
        reg=float(reg),
        solver=str(solver),
        pinv_rcond=float(pinv_rcond),
    )
    cfg["time_mode"] = "real"
    cfg["initialization"] = init_mode
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
            if record is not None:
                res = record["result"]
                if "final_params" not in res or "final_params_shape" not in res:
                    raise KeyError(
                        "Cached VarQRTE record is missing final parameters. "
                        "Re-run with force=True to refresh the cache."
                    )
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
            ansatz=str(ansatz_name),
            step_label="Time Step",
            seed=int(seed),
            dep_prob=0.0,
            amp_prob=0.0,
            noise_type=None,
            show=bool(show),
            save=True,
        )

    params_arr = np.array(params)
    result = {
        "molecule": str(molecule_label),
        "mapping": str(mapping_out),
        "unit": str(unit_out),
        "charge": int(charge_out),
        "basis": str(basis_out),
        "active_electrons": resolved_active_electrons,
        "active_orbitals": resolved_active_orbitals,
        "ansatz": str(ansatz_name),
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
            "pinv_rcond": float(pinv_rcond),
        },
    }

    record = {"config": cfg, "result": result}
    if cache_enabled and prefix is not None:
        save_run_record(prefix, record)
        print(f"\n💾 Saved run record: {prefix}.json\n")

    return result
