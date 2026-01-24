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

import json
import os
from typing import Any, Dict

from pennylane import numpy as np

from qite.engine import (
    build_ansatz as engine_build_ansatz,
)
from qite.engine import (
    make_device,
    make_energy_qnode,
    make_state_qnode,
    qite_step,
)
from qite.hamiltonian import build_hamiltonian
from qite.io_utils import (
    RESULTS_DIR,
    ensure_dirs,
    is_effectively_noisy,
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
    mapping: str = "jordan_wigner",
    unit: str = "angstrom",
    show: bool = True,
    # VarQITE numerics
    fd_eps: float = 1e-3,
    reg: float = 1e-6,
    solver: str = "solve",
    pinv_rcond: float = 1e-10,
    # Explicitly not supported, kept only to make intent unambiguous at callsites
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model=None,
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

    # Decide effective noisiness (canonical: affects device/filenames/caching)
    effective_noisy = is_effectively_noisy(
        bool(noisy),
        float(depolarizing_prob),
        float(amplitude_damping_prob),
        noise_model=noise_model,
    )
    if effective_noisy:
        raise ValueError(
            "VarQITE requires a pure statevector and is not supported with "
            "noisy/mixed-state simulation. Use the CLI's eval-noise mode for "
            "post-evaluation under noise."
        )

    # --- Hamiltonian pipeline (single source of truth) ---
    (
        H,
        qubits,
        hf_state,
        symbols,
        coordinates,
        basis,
        charge,
        mapping_out,
        unit_out,
    ) = build_hamiltonian(
        str(molecule),
        mapping=str(mapping),
        unit=str(unit),
    )
    basis = str(basis).lower()

    # --- Configuration & caching ---
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=np.array(coordinates, dtype=float),
        basis=str(basis),
        seed=int(seed),
        mapping=str(mapping_out),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        dtau=float(dtau),
        steps=int(steps),
        molecule_label=str(molecule),
        ansatz_desc=str(ansatz_name),
        noise_model_name=None,
        # VarQITE numerics (must be part of cache key)
        fd_eps=float(fd_eps),
        reg=float(reg),
        solver=str(solver),
        pinv_rcond=float(pinv_rcond),
        # These are stable metadata but not used by VarQITE math; keep explicit
        unit=str(unit_out),
        charge=int(charge),
    )

    sig = run_signature(cfg)
    prefix = make_filename_prefix(
        cfg, noisy=False, seed=int(seed), hash_str=sig, algo="varqite"
    )
    result_path = os.path.join(RESULTS_DIR, f"{prefix}.json")

    if not force and os.path.exists(result_path):
        print(f"\n📂 Found cached result: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            record = json.load(f)
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
        symbols=symbols,
        coordinates=np.array(coordinates, dtype=float),
        basis=str(basis),
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
        noise_model=None,
    )

    state_qnode = make_state_qnode(
        dev,
        ansatz_fn,
        int(qubits),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
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
            molecule=str(molecule),
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
        "molecule": str(molecule),
        "mapping": str(mapping_out),
        "unit": str(unit_out),
        "charge": int(charge),
        "basis": str(basis),
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
    save_run_record(prefix, record)
    print(f"\n💾 Saved run record to {result_path}\n")

    return result
