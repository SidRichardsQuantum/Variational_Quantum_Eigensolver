"""
vqe.eom_qse
-----------

Equation-of-Motion in an operator manifold (EOM-QSE): a *true* commutator-based
EOM formulation built on an operator pool {O_i} around a converged VQE reference.

This is intentionally **not** projection QSE (which solves Hc = ESc in the span
{O_i|psi>}).

Instead we build the EOM (commutator) matrix:
    A_ij = <psi| O_i^† [H, O_j] |psi>
         = <psi| O_i^† H O_j |psi> - <psi| O_i^† O_j H |psi>

and the overlap matrix:
    S_ij = <psi| O_i^† O_j |psi>

Then solve the generalized (generally non-Hermitian) eigenproblem:
    A c = ω S c

Excitation energies are ω; excited-state energies reported are E0 + ω where E0
is the VQE reference energy.

Stage-1: noiseless-only (statevector reference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml

from common.molecules import get_molecule_config

from .core import run_vqe
from .hamiltonian import build_hamiltonian
from .io_utils import (
    ensure_dirs,
    load_run_record,
    make_filename_prefix,
    make_run_config_dict,
    run_signature,
    save_run_record,
)

# ================================================================
# JSON-safe helpers
# ================================================================


def _json_safe_complex_matrix(M: np.ndarray, *, tol: float = 1e-14):
    """
    Encode a complex matrix for JSON:
      - if imag ~ 0 -> float(real)
      - else        -> [real, imag]
    """
    out = []
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


# ================================================================
# Operator specs (JSON-safe)
# ================================================================


@dataclass(frozen=True)
class PauliWordSpec:
    """
    A Pauli word acting on an explicit wire set.

    word: string in {'I','X','Y','Z'}^n for the given wires ordering.
    wires: list of wire indices aligned with positions in word.
    """

    word: str
    wires: Tuple[int, ...]

    def to_json(self) -> Dict[str, Any]:
        return {
            "type": "pauli_word",
            "word": str(self.word),
            "wires": [int(w) for w in self.wires],
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "PauliWordSpec":
        if d.get("type") != "pauli_word":
            raise ValueError(f"Unsupported operator spec type: {d.get('type')!r}")
        word = str(d.get("word", ""))
        wires = tuple(int(x) for x in d.get("wires", []))
        if not word or len(word) != len(wires):
            raise ValueError("Invalid pauli_word spec: word must match wires length.")
        if any(ch not in {"I", "X", "Y", "Z"} for ch in word):
            raise ValueError(f"Invalid pauli_word: {word!r}")
        return PauliWordSpec(word=word, wires=wires)


def _pauli_op_from_spec(spec: PauliWordSpec):
    """Construct a PennyLane observable from a PauliWordSpec."""
    ops = []
    for ch, w in zip(spec.word, spec.wires):
        if ch == "I":
            ops.append(qml.Identity(w))
        elif ch == "X":
            ops.append(qml.PauliX(w))
        elif ch == "Y":
            ops.append(qml.PauliY(w))
        elif ch == "Z":
            ops.append(qml.PauliZ(w))
        else:  # pragma: no cover
            raise ValueError(f"Unexpected Pauli char {ch!r}")
    if len(ops) == 0:
        raise ValueError("Empty PauliWordSpec is not allowed.")
    if len(ops) == 1:
        return ops[0]
    return qml.prod(*ops)


def _extract_hamiltonian_terms(H):
    """
    Return (coeffs, ops) from a PennyLane Hamiltonian-like object.

    Supports:
      - qml.Hamiltonian: .coeffs / .ops
      - Operator algebra: .terms() or .terms
    """
    if hasattr(H, "coeffs") and hasattr(H, "ops"):
        coeffs = list(getattr(H, "coeffs"))
        ops = list(getattr(H, "ops"))
        return coeffs, ops

    if hasattr(H, "terms"):
        t = getattr(H, "terms")
        try:
            if callable(t):
                coeffs, ops = t()
            else:
                coeffs, ops = t
            return list(coeffs), list(ops)
        except Exception:
            pass

    raise TypeError(
        "Hamiltonian does not expose .coeffs/.ops nor a usable .terms() interface."
    )


def _extract_pauli_word_spec(op, *, num_wires: int) -> Optional[PauliWordSpec]:
    """
    Best-effort extraction of a Pauli word spec from a Hamiltonian term op.

    Assumes qubit Hamiltonians built via qchem.molecular_hamiltonian
    are sums of Pauli products (X/Y/Z/Identity).
    """

    def _factor_to_wc(f) -> Optional[Tuple[int, str]]:
        name = getattr(f, "name", None)
        if name in {"PauliX", "PauliY", "PauliZ", "Identity"}:
            wires_list = list(getattr(f, "wires", []))
            if len(wires_list) == 0:
                return None
            w = int(wires_list[0])
            ch = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}[name]
            return w, ch
        return None

    def _flatten_factors(obj) -> List[Any]:
        if hasattr(obj, "operands"):
            out = []
            for child in list(obj.operands):
                out.extend(_flatten_factors(child))
            return out
        if hasattr(obj, "obs"):
            out = []
            for child in list(obj.obs):
                out.extend(_flatten_factors(child))
            return out
        return [obj]

    factors: Optional[List[Any]] = None

    if hasattr(op, "operands"):
        try:
            factors = list(op.operands)
        except Exception:
            factors = None

    if factors is None and hasattr(op, "obs"):
        try:
            factors = list(op.obs)
        except Exception:
            factors = None

    if factors is None:
        factors = _flatten_factors(op)

    if factors is None:
        wc = _factor_to_wc(op)
        if wc is None:
            return None
        factors = [op]

    wc_map: Dict[int, str] = {}
    for f in factors:
        wc = _factor_to_wc(f)
        if wc is None:
            return None
        w, ch = wc
        if w in wc_map:
            return None
        wc_map[w] = ch

    wires = tuple(sorted(wc_map.keys()))
    if any((w < 0 or w >= int(num_wires)) for w in wires):
        return None

    word = "".join(wc_map[w] for w in wires)
    return PauliWordSpec(word=word, wires=wires)


def _build_hamiltonian_topk_pool(
    H,
    *,
    num_wires: int,
    max_ops: int,
    include_identity: bool = True,
) -> List[PauliWordSpec]:
    """
    Build an operator pool from the Hamiltonian terms:
    take the top-|coeff| Pauli terms (deduplicated).
    """
    coeffs, ops = _extract_hamiltonian_terms(H)

    pairs = list(zip(list(coeffs), list(ops)))
    pairs.sort(key=lambda t: float(abs(t[0])), reverse=True)

    out: List[PauliWordSpec] = []
    seen: set[Tuple[str, Tuple[int, ...]]] = set()

    if include_identity:
        wires = tuple(range(int(num_wires)))
        out.append(PauliWordSpec(word="I" * int(num_wires), wires=wires))
        seen.add((out[-1].word, out[-1].wires))

    for c, op in pairs:
        if len(out) >= int(max_ops):
            break

        spec = _extract_pauli_word_spec(op, num_wires=int(num_wires))
        if spec is None:
            continue

        key = (spec.word, spec.wires)
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)

    if len(out) < 1:
        raise RuntimeError("Failed to build any operators for EOM-QSE pool.")
    return out


# ================================================================
# Linear algebra: generalized EVP with S filtering (non-Hermitian A)
# ================================================================


def _solve_generalized_evp_nonhermitian(
    A: np.ndarray,
    S: np.ndarray,
    *,
    eps: float,
    k: int,
    imag_tol: float,
    omega_eps: float,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Solve A c = ω S c where:
      - S is Hermitian positive semidefinite (overlap matrix)
      - A is generally non-Hermitian (commutator EOM matrix)

    We whiten using the S-eigendecomposition and then solve a standard
    (generally non-Hermitian) eigenproblem on the reduced space.

    Returns:
      - excitations: smallest-k positive real(ish) ω with |Im(ω)| <= imag_tol
      - diagnostics (including complex/negative modes statistics)
    """
    # Hermitize S defensively
    S = 0.5 * (S + S.conj().T)

    s_evals, s_evecs = np.linalg.eigh(S)
    s_evals = np.real(s_evals)

    keep = np.where(s_evals > float(eps))[0]
    if keep.size == 0:
        raise RuntimeError(
            f"All overlap eigenvalues are <= eps={eps}. "
            "Try increasing pool size or lowering eps."
        )

    Uk = s_evecs[:, keep]
    sk = s_evals[keep]

    X = Uk @ np.diag(1.0 / np.sqrt(sk))  # m x r

    # Reduced standard (non-Hermitian) eigenproblem: At = X^† A X
    At = X.conj().T @ A @ X

    # General eigensolver (non-Hermitian)
    w = np.linalg.eigvals(At)

    # Sort by real part
    w_sorted = w[np.argsort(np.real(w))]

    # Select physical-ish excitations: positive real, small imaginary part
    real_mask = np.abs(np.imag(w_sorted)) <= float(imag_tol)
    pos_mask = np.real(w_sorted) > float(omega_eps)
    good = w_sorted[real_mask & pos_mask]

    good_real = np.real(good)
    good_real_sorted = np.sort(good_real)

    k_eff = int(max(1, k))
    excitations = [float(x) for x in good_real_sorted[:k_eff]]

    diag: Dict[str, Any] = {
        "eps": float(eps),
        "imag_tol": float(imag_tol),
        "omega_eps": float(omega_eps),
        "subspace_dim": int(A.shape[0]),
        "kept_rank": int(keep.size),
        "S_eigs": [float(x) for x in s_evals.tolist()],
        "S_eig_min_kept": float(sk.min()),
        "S_eig_max_kept": float(sk.max()),
        "S_condition_kept": float((sk.max() / sk.min()) if sk.min() > 0 else np.inf),
        "num_eigs_total_reduced": int(w_sorted.size),
        "num_eigs_realish": int(np.count_nonzero(real_mask)),
        "num_eigs_positive_realish": int(good_real.size),
        "min_real_part": float(np.min(np.real(w_sorted))),
        "max_real_part": float(np.max(np.real(w_sorted))),
        "max_abs_imag_part": float(np.max(np.abs(np.imag(w_sorted)))),
    }

    # If none found, provide a helpful failure mode.
    if len(excitations) == 0:
        raise RuntimeError(
            "EOM-QSE found no positive excitation energies with "
            f"|Im(ω)| <= imag_tol={imag_tol} and Re(ω) > omega_eps={omega_eps}. "
            "This can occur if the reference is poor, the pool is too small, "
            "or the commutator matrix is strongly non-normal. "
            "Try increasing max_ops, relaxing imag_tol, or reducing eps."
        )

    return excitations, diag


# ================================================================
# Main entrypoint
# ================================================================


def run_eom_qse(
    molecule: str = "H2",
    *,
    k: int = 3,
    # Reference (VQE) controls
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    steps: int = 50,
    stepsize: float = 0.2,
    seed: int = 0,
    mapping: str = "jordan_wigner",
    # Operator pool controls
    pool: str = "hamiltonian_topk",
    max_ops: int = 24,
    operators: Optional[Sequence[Dict[str, Any]]] = None,
    # Generalized EVP controls
    eps: float = 1e-8,
    imag_tol: float = 1e-10,
    omega_eps: float = 1e-12,
    # Caching controls
    force: bool = False,
) -> Dict[str, Any]:
    """
    Run EOM-QSE (commutator EOM in an operator manifold) using a noiseless VQE reference.

    Notes
    -----
    - EOM-QSE is currently **noiseless-only**. It relies on the VQE reference statevector.
    - The reference VQE run is called with plot=False and noisy=False.

    Parameters
    ----------
    k
        Number of excitation energies to return (lowest-k positive ω).
    pool
        Operator pool strategy. Supported:
          - "hamiltonian_topk"
    max_ops
        Maximum number of operators in the pool (including identity if used).
    operators
        Optional explicit operator list (JSON specs). If provided, it overrides pool.
        Each operator spec must be {"type":"pauli_word","word":"...","wires":[...]}.
    eps
        Cutoff on overlap eigenvalues (discard small directions).
    imag_tol
        Imaginary-part tolerance for accepting an eigenvalue as "real-ish".
    omega_eps
        Require Re(ω) > omega_eps for acceptance as a physical excitation.
    """
    ensure_dirs()

    mol = str(molecule).strip()
    mapping_norm = str(mapping).strip().lower()
    pool_norm = str(pool).strip().lower()

    # ------------------------------------------------------------
    # 1) Reference VQE run (noiseless)
    # ------------------------------------------------------------
    vqe_res = run_vqe(
        molecule=mol,
        seed=int(seed),
        steps=int(steps),
        stepsize=float(stepsize),
        plot=False,
        ansatz_name=str(ansatz_name),
        optimizer_name=str(optimizer_name),
        noisy=False,
        mapping=mapping_norm,
        force=bool(force),
    )

    E0 = float(vqe_res["energy"])

    psi = np.array(vqe_res["final_state_real"], dtype=float) + 1j * np.array(
        vqe_res["final_state_imag"], dtype=float
    )
    nrm = np.linalg.norm(psi)
    if not np.isfinite(nrm) or nrm <= 0:
        raise RuntimeError("VQE reference state has invalid norm.")
    psi = psi / nrm

    num_wires = int(vqe_res["num_qubits"])

    # ------------------------------------------------------------
    # 2) Hamiltonian (for commutator matrix construction)
    # ------------------------------------------------------------
    H, n_qubits, *_ = build_hamiltonian(mol, mapping=mapping_norm, unit="angstrom")
    if int(n_qubits) != int(num_wires):
        raise RuntimeError(
            f"EOM-QSE: mismatch in qubit count (VQE={num_wires}, H={n_qubits})."
        )

    # ------------------------------------------------------------
    # 3) Operator pool
    # ------------------------------------------------------------
    op_specs: List[PauliWordSpec]
    if operators is not None:
        op_specs = [PauliWordSpec.from_json(dict(d)) for d in operators]
    else:
        if pool_norm != "hamiltonian_topk":
            raise ValueError(
                "Unsupported EOM-QSE pool. Supported: 'hamiltonian_topk' "
                f"(got {pool!r})."
            )
        op_specs = _build_hamiltonian_topk_pool(
            H,
            num_wires=int(num_wires),
            max_ops=int(max_ops),
            include_identity=True,
        )

    # ------------------------------------------------------------
    # 4) Config + caching
    # ------------------------------------------------------------
    cfg_mol = get_molecule_config(mol)
    symbols = list(cfg_mol["symbols"])
    coordinates = np.array(cfg_mol["coordinates"], dtype=float)
    basis = str(cfg_mol["basis"])

    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=str(basis),
        ansatz_desc=str(ansatz_name),
        optimizer_name=str(optimizer_name),
        stepsize=float(stepsize),
        max_iterations=int(steps),
        seed=int(seed),
        mapping=mapping_norm,
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        molecule_label=mol,
    )

    cfg["eom_qse_pool"] = pool_norm
    cfg["eom_qse_k"] = int(k)
    cfg["eom_qse_max_ops"] = int(max_ops)
    cfg["eom_qse_eps"] = float(eps)
    cfg["eom_qse_imag_tol"] = float(imag_tol)
    cfg["eom_qse_omega_eps"] = float(omega_eps)
    cfg["eom_qse_ops"] = [s.to_json() for s in op_specs]

    cfg["vqe_reference"] = {
        "ansatz": str(ansatz_name),
        "optimizer": str(optimizer_name),
        "steps": int(steps),
        "stepsize": float(stepsize),
        "seed": int(seed),
        "mapping": mapping_norm,
    }

    sig = run_signature(cfg)
    prefix = make_filename_prefix(
        cfg,
        noisy=False,
        seed=int(seed),
        hash_str=sig,
        algo="eom_qse",
    )

    if not force:
        record = load_run_record(prefix)
        if record is not None:
            return record["result"]

    # ------------------------------------------------------------
    # 5) Build commutator EOM matrix A and overlap S
    # ------------------------------------------------------------
    wire_order = list(range(int(num_wires)))
    Hmat = np.array(qml.matrix(H, wire_order=wire_order), dtype=complex)

    # Precompute matrices and action on vectors
    Omats: List[np.ndarray] = []
    Opsi: List[np.ndarray] = []
    for spec in op_specs:
        Oop = _pauli_op_from_spec(spec)
        Omat = np.array(qml.matrix(Oop, wire_order=wire_order), dtype=complex)
        Omats.append(Omat)
        Opsi.append(Omat @ psi)

    m = len(Opsi)
    S = np.zeros((m, m), dtype=complex)
    A = np.zeros((m, m), dtype=complex)

    # Helpful intermediates:
    #   v_j      = O_j |psi>
    #   Hv_j     = H (O_j |psi>)
    #   w        = H |psi>
    #   O_j w    = O_j (H |psi>)
    Hv = [Hmat @ vj for vj in Opsi]
    w = Hmat @ psi
    Ow = [Omats[j] @ w for j in range(m)]

    # Compute:
    #   S_ij = <O_i psi| O_j psi> = vdot(v_i, v_j)
    #   A_ij = <O_i psi| H |O_j psi> - <O_i psi| O_j H |psi>
    #        = vdot(v_i, Hv_j)     - vdot(v_i, O_j w)
    for i in range(m):
        vi = Opsi[i]
        for j in range(m):
            S[i, j] = np.vdot(vi, Opsi[j])
            A[i, j] = np.vdot(vi, Hv[j]) - np.vdot(vi, Ow[j])

    # ------------------------------------------------------------
    # 6) Solve generalized non-Hermitian EVP A c = ω S c
    # ------------------------------------------------------------
    omegas, diag = _solve_generalized_evp_nonhermitian(
        A,
        S,
        eps=float(eps),
        k=int(k),
        imag_tol=float(imag_tol),
        omega_eps=float(omega_eps),
    )
    energies = [float(E0 + w) for w in omegas]

    # ------------------------------------------------------------
    # 7) Save
    # ------------------------------------------------------------
    result: Dict[str, Any] = {
        "num_qubits": int(num_wires),
        "reference_energy": float(E0),
        "excitations": [float(x) for x in omegas],
        "eigenvalues": [float(x) for x in energies],
        "diagnostics": {
            **diag,
            "approx": "operator_commutator",
        },
        "config": cfg,
        "matrices": {
            "A": _json_safe_complex_matrix(A),
            "S": _json_safe_complex_matrix(S),
        },
    }

    save_run_record(prefix, {"config": cfg, "result": result})
    print(f"\n💾 Saved EOM-QSE run record: results/vqe/{prefix}.json\n")

    return result
