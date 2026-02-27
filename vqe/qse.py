"""
vqe.qse
-------

Quantum Subspace Expansion (QSE) for excited states (post-VQE).

High-level idea
---------------
1) Obtain a reference state |psi> from a converged VQE run (cached via run_vqe).
2) Choose a small operator set {O_i} spanning a local subspace around |psi>.
3) Build subspace matrices:
       H_ij = <psi| O_i^\dag H O_j |psi>
       S_ij = <psi| O_i^\dag   O_j |psi>
4) Solve the generalized eigenproblem: H c = E S c, yielding approximate
   ground + excited energies in the expanded subspace.
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
# Operator specs (JSON-safe)
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
    # Case 1: classic qml.Hamiltonian
    if hasattr(H, "coeffs") and hasattr(H, "ops"):
        coeffs = list(getattr(H, "coeffs"))
        ops = list(getattr(H, "ops"))
        return coeffs, ops

    # Case 2: newer operator objects often expose .terms() -> (coeffs, ops)
    if hasattr(H, "terms"):
        t = getattr(H, "terms")
        try:
            # terms() callable
            if callable(t):
                coeffs, ops = t()
            else:
                # terms property-like
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

    We assume qubit Hamiltonians built via qchem.molecular_hamiltonian
    are sums of Pauli products (X/Y/Z/Identity).
    """

    # Helper: map a single factor to (wire, char) if possible.
    def _factor_to_wc(f) -> Optional[Tuple[int, str]]:
        name = getattr(f, "name", None)
        if name in {"PauliX", "PauliY", "PauliZ", "Identity"}:
            # Some PennyLane operator objects (notably Identity in newer algebra)
            # can have empty wires for "global identity"/scalar terms.
            wires_list = list(getattr(f, "wires", []))
            if len(wires_list) == 0:
                return None  # treat as non-factor / scalar identity
            w = int(wires_list[0])
            ch = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}[name]
            return w, ch

        return None

    def _flatten_factors(obj) -> List[Any]:
        # Recursively peel off scalars and products.
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
        # Base case: leaf operator
        return [obj]

    # Gather factors for Tensor/Prod-like objects
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

    # If it's a single Pauli op, treat it as a one-factor product
    if factors is None:
        wc = _factor_to_wc(op)
        if wc is None:
            return None
        factors = [op]

    # Build wire->char map
    wc_map: Dict[int, str] = {}
    for f in factors:
        wc = _factor_to_wc(f)
        if wc is None:
            return None
        w, ch = wc
        # If duplicates occur, bail out (unexpected for Pauli products)
        if w in wc_map:
            return None
        wc_map[w] = ch

    # Represent on the *active wires only* (sorted wires present in product)
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
    # Sort by absolute coefficient magnitude, descending
    pairs.sort(key=lambda t: float(abs(t[0])), reverse=True)

    out: List[PauliWordSpec] = []
    seen: set[Tuple[str, Tuple[int, ...]]] = set()

    if include_identity:
        # Identity on all wires is a sane baseline.
        # Represent as I...I on wires [0..n-1].
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

    skipped = 0
    for c, op in pairs:
        spec = _extract_pauli_word_spec(op, num_wires=int(num_wires))
        if spec is None:
            skipped += 1
            continue
    print(
        f"QSE pool build: kept {len(out)} ops, skipped {skipped} of {len(pairs)} terms"
    )

    if len(out) < 1:
        raise RuntimeError("Failed to build any operators for QSE pool.")
    return out


# ================================================================
# Linear algebra: generalized eigenproblem with S filtering
# ================================================================


def _solve_generalized_evp(
    Hs: np.ndarray,
    Ss: np.ndarray,
    *,
    eps: float,
    k: int,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Solve H c = E S c with a stable whitening transform and S-eigenvalue cutoff.

    Returns (eigenvalues, diagnostics).
    """
    # Hermitize defensively (numerical noise)
    Hs = 0.5 * (Hs + Hs.conj().T)
    Ss = 0.5 * (Ss + Ss.conj().T)

    # Eigendecompose S (Hermitian)
    s_evals, s_evecs = np.linalg.eigh(Ss)
    s_evals = np.real(s_evals)

    # Keep modes above eps
    keep = np.where(s_evals > float(eps))[0]
    if keep.size == 0:
        raise RuntimeError(
            f"All overlap eigenvalues are <= eps={eps}. "
            "Try increasing pool size or lowering eps."
        )

    Uk = s_evecs[:, keep]
    sk = s_evals[keep]

    # Whitening transform X = Uk * diag(1/sqrt(sk))
    inv_sqrt = np.diag(1.0 / np.sqrt(sk))
    X = Uk @ inv_sqrt

    # Reduced standard EVP: H_tilde = X^† H X
    Ht = X.conj().T @ Hs @ X
    Ht = 0.5 * (Ht + Ht.conj().T)

    evals, _ = np.linalg.eigh(Ht)
    evals = np.real(evals)
    evals_sorted = np.sort(evals)

    k_eff = int(max(1, k))
    evals_out = [float(x) for x in evals_sorted[:k_eff]]

    diag: Dict[str, Any] = {
        "eps": float(eps),
        "subspace_dim": int(Hs.shape[0]),
        "kept_rank": int(keep.size),
        "S_eigs": [float(x) for x in s_evals.tolist()],
        "S_eig_min_kept": float(sk.min()),
        "S_eig_max_kept": float(sk.max()),
        "S_condition_kept": float((sk.max() / sk.min()) if sk.min() > 0 else np.inf),
    }
    return evals_out, diag


# ================================================================
# Main entrypoint
# ================================================================


def run_qse(
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
    # QSE pool controls
    pool: str = "hamiltonian_topk",
    max_ops: int = 24,
    operators: Optional[Sequence[Dict[str, Any]]] = None,
    eps: float = 1e-8,
    # Caching controls
    force: bool = False,
):
    """
    Run Quantum Subspace Expansion (QSE) using a cached/noiseless VQE reference.

    Notes
    -----
    - QSE is currently **noiseless-only**. It relies on the VQE reference statevector.
    - The reference VQE run is called with plot=False and noisy=False.

    Parameters
    ----------
    k
        Number of eigenvalues to return (lowest-k).
    pool
        Operator pool strategy. Currently supported:
          - "hamiltonian_topk"
    max_ops
        Maximum number of operators in the pool (including identity if used).
    operators
        Optional explicit operator list (JSON specs). If provided, it overrides pool.
        Each operator spec must be {"type":"pauli_word","word":"...","wires":[...]}.
    eps
        Cutoff threshold on overlap matrix eigenvalues (discard small directions).
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

    psi = np.array(vqe_res["final_state_real"], dtype=float) + 1j * np.array(
        vqe_res["final_state_imag"], dtype=float
    )
    # Normalise defensively
    nrm = np.linalg.norm(psi)
    if not np.isfinite(nrm) or nrm <= 0:
        raise RuntimeError("VQE reference state has invalid norm.")
    psi = psi / nrm

    num_wires = int(vqe_res["num_qubits"])

    # ------------------------------------------------------------
    # 2) Hamiltonian (for QSE subspace matrix construction)
    # ------------------------------------------------------------
    H, n_qubits, *_ = build_hamiltonian(mol, mapping=mapping_norm, unit="angstrom")
    if int(n_qubits) != int(num_wires):
        raise RuntimeError(
            f"QSE: mismatch in qubit count (VQE={num_wires}, H={n_qubits})."
        )

    # ------------------------------------------------------------
    # 3) Operator pool
    # ------------------------------------------------------------
    op_specs: List[PauliWordSpec]
    if operators is not None:
        # Explicit operator list
        op_specs = [PauliWordSpec.from_json(dict(d)) for d in operators]
    else:
        if pool_norm != "hamiltonian_topk":
            raise ValueError(
                f"Unsupported QSE pool. Supported: 'hamiltonian_topk' (got {pool!r})."
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

    cfg["qse_pool"] = pool_norm
    cfg["qse_k"] = int(k)
    cfg["qse_max_ops"] = int(max_ops)
    cfg["qse_eps"] = float(eps)
    cfg["qse_ops"] = [s.to_json() for s in op_specs]

    # Include a compact reference summary (useful for debugging)
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
        algo="qse",
    )

    if not force:
        record = load_run_record(prefix)
        if record is not None:
            return record["result"]

    # ------------------------------------------------------------
    # 5) Build subspace matrices Hs and Ss
    # ------------------------------------------------------------
    Hmat = np.array(qml.matrix(H), dtype=complex)

    # Precompute O_i|psi>
    wire_order = list(range(int(num_wires)))

    Opsi: List[np.ndarray] = []
    for spec in op_specs:
        A = _pauli_op_from_spec(spec)
        Omat = np.array(qml.matrix(A, wire_order=wire_order), dtype=complex)
        Opsi.append(Omat @ psi)

    m = len(Opsi)
    Hs = np.zeros((m, m), dtype=complex)
    Ss = np.zeros((m, m), dtype=complex)

    # Compute H_ij = <O_i psi| H |O_j psi>, S_ij = <O_i psi|O_j psi>
    H_Opsi = [Hmat @ v for v in Opsi]
    for i in range(m):
        vi = Opsi[i]
        for j in range(m):
            Hs[i, j] = np.vdot(vi, H_Opsi[j])
            Ss[i, j] = np.vdot(vi, Opsi[j])

    # ------------------------------------------------------------
    # 6) Solve generalized EVP
    # ------------------------------------------------------------
    evals, diag = _solve_generalized_evp(Hs, Ss, eps=float(eps), k=int(k))

    # ------------------------------------------------------------
    # 7) Save
    # ------------------------------------------------------------
    result = {
        "eigenvalues": [float(x) for x in evals],
        "H_subspace": _json_safe_complex_matrix(Hs),
        "S_subspace": _json_safe_complex_matrix(Ss),
        "diagnostics": diag,
        "num_qubits": int(num_wires),
        "config": cfg,
    }

    save_run_record(prefix, {"config": cfg, "result": result})
    print(f"\n💾 Saved QSE run record: results/vqe/{prefix}.json\n")

    return result
