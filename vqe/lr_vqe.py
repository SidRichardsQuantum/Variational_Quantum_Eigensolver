"""
vqe.lr_vqe
---------
Linear-Response VQE (LR-VQE) for excited states.

True Stage-1 implementation: tangent-space TDA generalized EVP.

Given converged VQE parameters θ* and reference state |ψ(θ*)⟩ with energy E0:

    |∂i⟩ ≈ (|ψ(θ*+δ e_i)⟩ - |ψ(θ*-δ e_i)⟩) / (2δ)

Metric:
    S_ij = ⟨∂i|∂j⟩

Response (TDA):
    A_ij = ⟨∂i|(H - E0)|∂j⟩

Solve:
    A c = ω S c

Return ω_k and energies E0 + ω_k.

Noiseless only (statevector reference).
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pennylane as qml

from .core import run_vqe
from .engine import build_ansatz as engine_build_ansatz
from .engine import make_device, make_state_qnode
from .hamiltonian import build_hamiltonian
from .io_utils import (
    ensure_dirs,
    load_run_record,
    make_filename_prefix,
    make_run_config_dict,
    run_signature,
    save_run_record,
)


def _json_safe_complex_matrix(M: np.ndarray, *, tol: float = 1e-14):
    out: list[list[float | list[float]]] = []
    for row in M.tolist():
        r = []
        for x in row:
            z = complex(x)
            if abs(z.imag) < float(tol):
                r.append(float(z.real))
            else:
                r.append([float(z.real), float(z.imag)])
        out.append(r)
    return out


def _solve_generalized_evp(
    A: np.ndarray,
    S: np.ndarray,
    *,
    eps: float,
    k: int,
) -> Tuple[List[float], Dict[str, Any]]:
    A = 0.5 * (A + A.conj().T)
    S = 0.5 * (S + S.conj().T)

    s_evals, s_evecs = np.linalg.eigh(S)
    s_evals = np.real(s_evals)

    keep = np.where(s_evals > float(eps))[0]
    if keep.size == 0:
        raise RuntimeError(
            f"All overlap eigenvalues are <= eps={eps}. "
            "Try lowering eps or increasing VQE convergence / fd_eps."
        )

    Uk = s_evecs[:, keep]
    sk = s_evals[keep]

    X = Uk @ np.diag(1.0 / np.sqrt(sk))
    At = X.conj().T @ A @ X
    At = 0.5 * (At + At.conj().T)

    w = np.linalg.eigvalsh(At)
    w = np.real(w)
    w_sorted = np.sort(w)

    k_eff = int(max(1, k))
    w_out = [float(x) for x in w_sorted[:k_eff]]

    diag: Dict[str, Any] = {
        "eps": float(eps),
        "subspace_dim": int(A.shape[0]),
        "kept_rank": int(keep.size),
        "S_eigs": [float(x) for x in s_evals.tolist()],
        "S_eig_min_kept": float(sk.min()),
        "S_eig_max_kept": float(sk.max()),
        "S_condition_kept": float((sk.max() / sk.min()) if sk.min() > 0 else np.inf),
    }
    return w_out, diag


def _state_to_ket(state_or_rho: np.ndarray) -> np.ndarray:
    st = np.asarray(state_or_rho)
    if st.ndim == 1:
        return st
    raise ValueError(
        "LR-VQE requires a statevector backend (noiseless). "
        "Got a density matrix; noisy LR is not implemented in this stage."
    )


def _exact_evals_from_hamiltonian(H: Any, n_qubits: int) -> np.ndarray:
    Hmat = np.asarray(
        qml.matrix(H, wire_order=list(range(int(n_qubits)))), dtype=complex
    )
    return np.sort(np.linalg.eigvalsh(Hmat).real)


def run_lr_vqe(
    molecule: str = "H2",
    *,
    k: int = 3,
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    steps: int = 50,
    stepsize: float = 0.2,
    seed: int = 0,
    mapping: str = "jordan_wigner",
    fd_eps: float = 1e-3,
    eps: float = 1e-10,
    force: bool = False,
    plot: bool = False,
    show: bool = True,
    save: bool = False,
) -> Dict[str, Any]:
    ensure_dirs()

    mapping_norm = str(mapping).strip().lower()
    molecule_label = str(molecule).strip()

    H, num_qubits, _hf_state, symbols, coordinates, basis, _charge, _unit = (
        build_hamiltonian(
            str(molecule),
            mapping=mapping_norm,
            unit="angstrom",
        )
    )

    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=str(basis).strip().lower(),
        ansatz_desc=str(ansatz_name),
        optimizer_name=str(optimizer_name),
        stepsize=float(stepsize),
        max_iterations=int(steps),
        seed=int(seed),
        mapping=mapping_norm,
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        molecule_label=molecule_label,
    )
    cfg["lr"] = {
        "k": int(k),
        "fd_eps": float(fd_eps),
        "eps": float(eps),
        "approx": "tangent_tda",
    }

    sig = run_signature(cfg)
    prefix = make_filename_prefix(
        cfg, noisy=False, seed=int(seed), hash_str=sig, algo="lr"
    )

    want_plot = bool(plot) or bool(save)
    show_plot = bool(show) or bool(save)

    if not force:
        record = load_run_record(prefix)
        if record is not None:
            res = record["result"]
            if want_plot:
                from .visualize import plot_lr_vqe_spectrum

                exact_evals = _exact_evals_from_hamiltonian(H, int(num_qubits))
                plot_lr_vqe_spectrum(
                    exact_evals,
                    res,
                    molecule_label=molecule_label,
                    show=show_plot,
                    save=bool(save),
                )
            return res

    vqe_res = run_vqe(
        molecule=str(molecule),
        seed=int(seed),
        steps=int(steps),
        stepsize=float(stepsize),
        plot=False,
        ansatz_name=str(ansatz_name),
        optimizer_name=str(optimizer_name),
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        force=True,
        mapping=mapping_norm,
    )

    theta_star = np.asarray(vqe_res["final_params"], dtype=float).ravel()
    if theta_star.size < 1:
        raise ValueError("LR-VQE requires at least 1 variational parameter.")

    E0 = float(vqe_res["energy"])

    dev = make_device(int(num_qubits), noisy=False)

    ansatz_fn, _ = engine_build_ansatz(
        str(ansatz_name),
        int(num_qubits),
        seed=int(seed),
        symbols=symbols,
        coordinates=coordinates,
        basis=str(basis).strip().lower(),
    )

    state_qnode = make_state_qnode(
        dev,
        ansatz_fn,
        int(num_qubits),
        noisy=False,
        symbols=symbols,
        coordinates=coordinates,
        basis=str(basis).strip().lower(),
    )

    p = int(theta_star.size)
    dpsi: list[np.ndarray] = []
    for i in range(p):
        e = np.zeros_like(theta_star)
        e[i] = 1.0
        psi_p = _state_to_ket(state_qnode(theta_star + float(fd_eps) * e))
        psi_m = _state_to_ket(state_qnode(theta_star - float(fd_eps) * e))
        dpsi.append((psi_p - psi_m) / (2.0 * float(fd_eps)))

    D = np.stack(dpsi, axis=1)

    Hmat = np.asarray(
        qml.matrix(H, wire_order=list(range(int(num_qubits)))), dtype=complex
    )
    Heff = Hmat - float(E0) * np.eye(Hmat.shape[0], dtype=complex)

    S = D.conj().T @ D
    A = D.conj().T @ (Heff @ D)

    S = 0.5 * (S + S.conj().T)
    A = 0.5 * (A + A.conj().T)

    omegas, diag = _solve_generalized_evp(A, S, eps=float(eps), k=int(k))
    energies = [float(E0 + w) for w in omegas]

    result: Dict[str, Any] = {
        "num_qubits": int(num_qubits),
        "reference_energy": float(E0),
        "excitations": omegas,
        "eigenvalues": energies,
        "diagnostics": {
            **diag,
            "fd_eps": float(fd_eps),
            "reference": {
                "ansatz": str(ansatz_name),
                "optimizer": str(optimizer_name),
                "steps": int(steps),
                "stepsize": float(stepsize),
                "seed": int(seed),
                "mapping": mapping_norm,
            },
        },
        "config": cfg,
        "matrices": {
            "S": _json_safe_complex_matrix(S),
            "A": _json_safe_complex_matrix(A),
        },
    }

    save_run_record(prefix, {"config": cfg, "result": result})

    if want_plot:
        from .visualize import plot_lr_vqe_spectrum

        exact_evals = _exact_evals_from_hamiltonian(H, int(num_qubits))
        plot_lr_vqe_spectrum(
            exact_evals,
            result,
            molecule_label=molecule_label,
            show=show_plot,
            save=bool(save),
        )

    return result
