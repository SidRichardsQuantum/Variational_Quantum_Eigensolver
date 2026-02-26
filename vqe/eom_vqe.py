"""
vqe.eom_vqe
-----------
Equation-of-Motion VQE (EOM-VQE) for excited states.

True Stage-1 implementation: tangent-space full-response (RPA-like) generalized EVP.

Given converged VQE parameters θ* and reference state |ψ(θ*)⟩ with energy E0:

    |∂i⟩ ≈ (|ψ(θ*+δ e_i)⟩ - |ψ(θ*-δ e_i)⟩) / (2δ)

Metric (tangent overlap):
    S_ij = ⟨∂i|∂j⟩

Full-response blocks (tangent-space EOM/RPA form):
    A_ij = ⟨∂i|(H - E0)|∂j⟩
    B_ij = ⟨∂i|(H - E0)|∂j*⟩

Solve the structured generalized eigenproblem:
    [ A   B ] [X] = ω [ S   0 ] [X]
    [-B* -A*] [Y]     [ 0 -S*] [Y]

Return excitation energies ω_k (positive roots) and excited energies E0 + ω_k.

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


def _state_to_ket(state_or_rho: np.ndarray) -> np.ndarray:
    st = np.asarray(state_or_rho)
    if st.ndim == 1:
        return st
    raise ValueError(
        "EOM-VQE requires a statevector backend (noiseless). "
        "Got a density matrix; noisy EOM is not implemented in this stage."
    )


def _exact_evals_from_hamiltonian(H: Any, n_qubits: int) -> np.ndarray:
    Hmat = np.asarray(
        qml.matrix(H, wire_order=list(range(int(n_qubits)))), dtype=complex
    )
    return np.sort(np.linalg.eigvalsh(Hmat).real)


def _symmetry_deviation_hermitian(M: np.ndarray) -> float:
    denom = float(np.linalg.norm(M)) + 1e-30
    return float(np.linalg.norm(M - M.conj().T) / denom)


def _symmetry_deviation_symmetric(M: np.ndarray) -> float:
    denom = float(np.linalg.norm(M)) + 1e-30
    return float(np.linalg.norm(M - M.T) / denom)


def _solve_full_response_eom(
    A: np.ndarray,
    B: np.ndarray,
    S: np.ndarray,
    *,
    eps: float,
    k: int,
    omega_eps: float = 1e-12,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Solve the tangent-space full-response EOM generalized EVP.

    Steps:
      1) Hermitianize S and A (numerical stability), keep B as-is but record symmetry.
      2) Filter by overlap eigenvalues of S > eps (same as LR-VQE).
      3) Orthonormalize the tangent basis via X = U diag(1/sqrt(s)).
      4) Build transformed blocks:
            At = X† A X
            Bt = X† B X*
      5) Convert generalized problem with metric diag(I, -I) to standard eigenproblem:
            K v = ω v, where K = M Rt and M = diag(I, -I)
         With Rt = [[At, Bt], [-Bt*, -At*]] we have:
            K = [[At, Bt], [Bt*, At*]]
      6) Hermitianize K for stability; eigenvalues appear in ± pairs.
      7) Return the smallest k positive eigenvalues as excitation energies.
    """
    A = 0.5 * (A + A.conj().T)
    S = 0.5 * (S + S.conj().T)

    # Diagnostics: symmetry of raw inputs (pre-projection)
    diag_sym: Dict[str, Any] = {
        "A_hermitian_deviation": _symmetry_deviation_hermitian(A),
        "S_hermitian_deviation": _symmetry_deviation_hermitian(S),
        "B_symmetric_deviation": _symmetry_deviation_symmetric(B),
        "B_hermitian_deviation": _symmetry_deviation_hermitian(B),
    }

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

    X = Uk @ np.diag(1.0 / np.sqrt(sk))  # p x r

    At = X.conj().T @ A @ X
    At = 0.5 * (At + At.conj().T)

    # Note: B couples |∂⟩ to |∂⟩*; right transform uses X*.
    Bt = X.conj().T @ B @ X.conj()
    # In ideal full-response EOM, Bt is symmetric; we symmetrize for numerical stability
    # but keep deviations in diagnostics.
    Bt_sym_dev = _symmetry_deviation_symmetric(Bt)
    Bt = 0.5 * (Bt + Bt.T)

    # Build K = [[At, Bt], [Bt*, At*]]
    K = np.block(
        [
            [At, Bt],
            [Bt.conj(), At.conj()],
        ]
    )
    K_herm_dev = _symmetry_deviation_hermitian(K)
    K = 0.5 * (K + K.conj().T)

    w = np.linalg.eigvalsh(K)
    w = np.real(w)
    w_sorted = np.sort(w)

    # Positive roots are the physical excitation energies (RPA spectrum has ± pairs).
    pos = w_sorted[w_sorted > float(omega_eps)]
    if pos.size == 0:
        raise RuntimeError(
            "EOM-VQE full-response solver found no positive excitation energies. "
            "This can indicate an unstable reference, insufficient convergence, "
            "or overly aggressive overlap filtering."
        )

    k_eff = int(max(1, k))
    w_out = [float(x) for x in pos[:k_eff]]

    diag: Dict[str, Any] = {
        "eps": float(eps),
        "omega_eps": float(omega_eps),
        "subspace_dim": int(A.shape[0]),
        "kept_rank": int(keep.size),
        "S_eigs": [float(x) for x in s_evals.tolist()],
        "S_eig_min_kept": float(sk.min()),
        "S_eig_max_kept": float(sk.max()),
        "S_condition_kept": float((sk.max() / sk.min()) if sk.min() > 0 else np.inf),
        "Bt_symmetric_deviation_projected": float(Bt_sym_dev),
        "K_hermitian_deviation_projected": float(K_herm_dev),
        "raw_eigs_min": float(w_sorted[0]),
        "raw_eigs_max": float(w_sorted[-1]),
        "num_pos_roots": int(pos.size),
        **diag_sym,
    }
    return w_out, diag


def run_eom_vqe(
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
    omega_eps: float = 1e-12,
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
    cfg["eom"] = {
        "k": int(k),
        "fd_eps": float(fd_eps),
        "eps": float(eps),
        "omega_eps": float(omega_eps),
        "approx": "tangent_full_response",
    }

    sig = run_signature(cfg)
    prefix = make_filename_prefix(
        cfg, noisy=False, seed=int(seed), hash_str=sig, algo="eom_vqe"
    )

    want_plot = bool(plot) or bool(save)
    show_plot = bool(show) or bool(save)

    if not force:
        record = load_run_record(prefix)
        if record is not None:
            res = record["result"]
            if want_plot:
                from .visualize import plot_eom_vqe_spectrum

                exact_evals = _exact_evals_from_hamiltonian(H, int(num_qubits))
                plot_eom_vqe_spectrum(
                    exact_evals,
                    res,
                    molecule_label=molecule_label,
                    show=show_plot,
                    save=bool(save),
                )
            return res

    # 1) Converge VQE reference (noiseless only).
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
        raise ValueError("EOM-VQE requires at least 1 variational parameter.")

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

    # 2) Build finite-difference tangent vectors.
    p = int(theta_star.size)
    dpsi: list[np.ndarray] = []
    for i in range(p):
        e = np.zeros_like(theta_star)
        e[i] = 1.0
        psi_p = _state_to_ket(state_qnode(theta_star + float(fd_eps) * e))
        psi_m = _state_to_ket(state_qnode(theta_star - float(fd_eps) * e))
        dpsi.append((psi_p - psi_m) / (2.0 * float(fd_eps)))

    D = np.stack(dpsi, axis=1)  # (dim, p)

    # 3) Hamiltonian in matrix form (statevector reference).
    Hmat = np.asarray(
        qml.matrix(H, wire_order=list(range(int(num_qubits)))), dtype=complex
    )
    Heff = Hmat - float(E0) * np.eye(Hmat.shape[0], dtype=complex)

    # 4) Tangent-space blocks.
    S = D.conj().T @ D
    A = D.conj().T @ (Heff @ D)

    # For the "conjugate tangent" coupling block, represent |∂⟩* in the computational basis
    # by elementwise conjugation of the tangent kets.
    D_star = D.conj()
    B = D.conj().T @ (Heff @ D_star)

    # Stabilize S/A numerically; keep B as computed (solver will symmetrize Bt after projection).
    S = 0.5 * (S + S.conj().T)
    A = 0.5 * (A + A.conj().T)

    omegas, diag = _solve_full_response_eom(
        A, B, S, eps=float(eps), k=int(k), omega_eps=float(omega_eps)
    )
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
            "B": _json_safe_complex_matrix(B),
        },
    }

    save_run_record(prefix, {"config": cfg, "result": result})

    if want_plot:
        from .visualize import plot_eom_vqe_spectrum

        exact_evals = _exact_evals_from_hamiltonian(H, int(num_qubits))
        plot_eom_vqe_spectrum(
            exact_evals,
            result,
            molecule_label=molecule_label,
            show=show_plot,
            save=bool(save),
        )

    return result
