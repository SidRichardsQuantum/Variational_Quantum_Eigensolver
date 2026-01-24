"""
qite.core
=========
High-level orchestration for imaginary-time / QITE-style workflows.

This module mirrors the ergonomics of vqe.core and qpe.core:

- A cached main entrypoint:        run_qite(...)
- Optional plotting + saving:      qite.visualize
- Reproducible I/O + hashing:      qite.io_utils
- Circuit plumbing / QNodes:       qite.engine

Important
---------
This module now runs a "true" VarQITE step via qite.engine.qite_step(...)
(McLachlan variational imaginary-time evolution). This requires a pure
statevector, so noisy/mixed-state runs are intentionally not supported.
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
from vqe.hamiltonian import build_hamiltonian as build_vqe_hamiltonian
from vqe.hamiltonian import hartree_fock_state


# ================================================================
# SHARED HELPERS
# ================================================================
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


# ================================================================
# MAIN QITE EXECUTION (CACHED)
# ================================================================
def run_qite(
    molecule: str = "H2",
    *,
    seed: int = 0,
    steps: int = 50,
    dtau: float = 0.2,
    plot: bool = True,
    ansatz_name: str = "UCCSD",
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model=None,
    force: bool = False,
    mapping: str = "jordan_wigner",
    show: bool = True,
    # VarQITE numerics
    fd_eps: float = 1e-3,
    reg: float = 1e-6,
    solver: str = "solve",
    pinv_rcond: float = 1e-10,
) -> Dict[str, Any]:
    """
    Run VarQITE end-to-end with caching.

    Notes
    -----
    VarQITE uses a McLachlan linear-system update requiring a pure statevector.
    Therefore noisy/mixed-state runs are intentionally not supported.

    Returns
    -------
    dict
        {
            "energy": float,
            "energies": [float, ...],
            "steps": int,
            "dtau": float,
            "final_state_real": [...],
            "final_state_imag": [...],
            "num_qubits": int,
        }
    """
    ensure_dirs()
    np.random.seed(int(seed))

    # --- Hamiltonian & molecular data ---
    H, qubits, symbols, coordinates, basis = build_vqe_hamiltonian(
        molecule, mapping=mapping
    )
    basis = str(basis).lower()

    # Decide effective noisiness (canonical: affects device, filenames, caching)
    effective_noisy = is_effectively_noisy(
        noisy,
        depolarizing_prob,
        amplitude_damping_prob,
        noise_model=noise_model,
    )

    if bool(effective_noisy):
        raise ValueError(
            "VarQITE (McLachlan) requires a pure statevector and is not currently "
            "supported with noisy/mixed-state simulation. Run with noisy=False."
        )

    # --- Configuration & caching ---
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        ansatz_desc=ansatz_name,
        seed=int(seed),
        mapping=mapping,
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        dtau=float(dtau),
        steps=int(steps),
        molecule_label=molecule,
        noise_model_name=None,
        # VarQITE numerics (must be part of cache key)
        fd_eps=float(fd_eps),
        reg=float(reg),
        solver=str(solver),
        pinv_rcond=float(pinv_rcond),
    )

    sig = run_signature(cfg)
    prefix = make_filename_prefix(
        cfg,
        noisy=False,
        seed=int(seed),
        hash_str=sig,
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
    dev = make_device(qubits, noisy=False)

    hf_state = hartree_fock_state(
        molecule,
        mapping=mapping,
    )

    ansatz_fn, params = engine_build_ansatz(
        ansatz_name,
        qubits,
        seed=int(seed),
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        requires_grad=True,
        hf_state=hf_state,
    )

    energy_qnode = make_energy_qnode(
        H,
        dev,
        ansatz_fn,
        qubits,
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        noise_model=None,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )

    state_qnode = make_state_qnode(
        dev,
        ansatz_fn,
        qubits,
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        noise_model=None,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
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
            molecule=molecule,
            method="VarQITE",
            ansatz=ansatz_name,
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
        "energy": final_energy,
        "energies": [float(e) for e in energies],
        "steps": int(steps),
        "dtau": float(dtau),
        "final_state_real": np.real(final_state).tolist(),
        "final_state_imag": np.imag(final_state).tolist(),
        "num_qubits": int(qubits),
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


# ================================================================
# MULTI-SEED NOISE STUDIES (QITE)
# ================================================================
def run_qite_multi_seed_noise(
    *,
    molecule: str = "H2",
    ansatz_name: str = "UCCSD",
    steps: int = 30,
    dtau: float = 0.2,
    seeds=None,
    noise_type: str = "depolarizing",
    noise_levels=None,
    mapping: str = "jordan_wigner",
    force: bool = False,
    show: bool = True,
):
    """
    Multi-seed noise statistics for QITE.

    VarQITE currently requires a pure statevector and therefore does not support
    noisy/mixed-state simulation. This entrypoint is intentionally disabled
    until we decide on a principled "noisy VarQITE" design.
    """
    raise NotImplementedError(
        "run_qite_multi_seed_noise is disabled for VarQITE because VarQITE currently "
        "requires a pure statevector (no noisy/mixed-state simulation)."
    )
