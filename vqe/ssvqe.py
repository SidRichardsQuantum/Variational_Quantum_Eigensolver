# vqe/ssvqe.py
from __future__ import annotations

import json
from typing import Callable, List, Optional, Sequence, Tuple

import pennylane as qml
from pennylane import numpy as np

from .ansatz import _build_ucc_data
from .engine import (
    _call_ansatz,
    apply_optional_noise,
    build_ansatz,
    build_optimizer,
    make_device,
)
from .hamiltonian import build_hamiltonian
from .io_utils import (
    RESULTS_DIR,
    ensure_dirs,
    make_filename_prefix,
    make_run_config_dict,
    run_signature,
    save_run_record,
)
from .visualize import plot_multi_state_convergence


def _apply_single_excitation_to_det(hf: np.ndarray, exc: Sequence[int]) -> list[int]:
    """Return determinant bitstring after applying a single excitation [i, a]."""
    if len(exc) != 2:
        raise ValueError(f"Single excitation must have length 2, got {exc}")
    i, a = map(int, exc)

    det = np.array(hf, dtype=int).copy()
    if det[i] != 1:
        raise ValueError(f"Invalid single excitation {exc}: orbital {i} not occupied.")
    if det[a] != 0:
        raise ValueError(f"Invalid single excitation {exc}: orbital {a} already occ.")

    det[i] = 0
    det[a] = 1
    return det.tolist()


def _apply_double_excitation_to_det(hf: np.ndarray, exc: Sequence[int]) -> list[int]:
    """Return determinant bitstring after applying a double excitation [i, j, a, b]."""
    if len(exc) != 4:
        raise ValueError(f"Double excitation must have length 4, got {exc}")
    i, j, a, b = map(int, exc)

    det = np.array(hf, dtype=int).copy()
    if det[i] != 1 or det[j] != 1:
        raise ValueError(f"Invalid double excitation {exc}: {i},{j} must be occupied.")
    if det[a] != 0 or det[b] != 0:
        raise ValueError(
            f"Invalid double excitation {exc}: {a},{b} must be unoccupied."
        )
    if len({i, j, a, b}) != 4:
        raise ValueError(f"Invalid double excitation {exc}: indices must be distinct.")

    det[i] = 0
    det[j] = 0
    det[a] = 1
    det[b] = 1
    return det.tolist()


def _ucc_reference_states_from_excitations(
    hf_state: Sequence[int],
    singles: Sequence[Sequence[int]],
    doubles: Sequence[Sequence[int]],
    *,
    num_states: int,
    include_doubles: bool = True,
) -> list[list[int]]:
    """
    Build a chemistry-aware orthogonal reference set:
        [ HF, HF->single_0, HF->single_1, ..., (optionally) HF->double_0, ... ]

    All are computational-basis determinants, hence orthogonal by construction.
    """
    hf = np.array(hf_state, dtype=int)
    refs: list[list[int]] = [hf.tolist()]

    for exc in singles:
        if len(refs) >= num_states:
            return refs
        try:
            refs.append(_apply_single_excitation_to_det(hf, exc))
        except ValueError:
            continue

    if include_doubles:
        for exc in doubles:
            if len(refs) >= num_states:
                return refs
            try:
                refs.append(_apply_double_excitation_to_det(hf, exc))
            except ValueError:
                continue

    # Fallback: HF + single bit flips (still orthogonal, but less “chemistry-aware”)
    if len(refs) < num_states:
        n = len(hf)
        for i in range(n):
            if len(refs) >= num_states:
                break
            s = hf.copy()
            s[i] = 1 - s[i]
            cand = s.tolist()
            if cand not in refs:
                refs.append(cand)

    return refs[:num_states]


def _default_reference_states(num_states: int, num_wires: int) -> List[List[int]]:
    """
    Default orthogonal computational-basis reference states:
        |0...0>, |0...01>, |0...10>, ...

    Note: for chemistry, you will usually want HF + excitations instead.
    """
    if num_states < 1:
        raise ValueError("num_states must be >= 1")
    if num_states > 2**num_wires:
        raise ValueError(
            f"num_states={num_states} exceeds Hilbert space size 2**{num_wires}"
        )

    states: List[List[int]] = []
    for k in range(num_states):
        bits = [(k >> (num_wires - 1 - i)) & 1 for i in range(num_wires)]
        states.append(bits)
    return states


def _validate_reference_states(
    reference_states: Sequence[Sequence[int]],
    *,
    num_states: int,
    num_wires: int,
) -> list[list[int]]:
    if len(reference_states) != num_states:
        raise ValueError(
            f"reference_states must have length num_states={num_states}, "
            f"got {len(reference_states)}"
        )

    refs = [list(map(int, s)) for s in reference_states]
    for i, s in enumerate(refs):
        if len(s) != num_wires:
            raise ValueError(
                f"reference_states[{i}] has length {len(s)} but num_wires={num_wires}"
            )
        if any(b not in (0, 1) for b in s):
            raise ValueError(
                f"reference_states[{i}] must be a bitstring of 0/1 values, got {s}"
            )

    if len({tuple(s) for s in refs}) != len(refs):
        raise ValueError(
            "reference_states contain duplicates; cannot enforce orthogonality."
        )

    return refs


def _compute_sorted_finals(
    energies_per_state: Sequence[Sequence[float]],
) -> Tuple[list[float], list[int], list[float]]:
    """
    Compute canonical energy-ranked finals without reordering trajectories.

    Returns
    -------
    final_energies_by_ref : list[float]
        Final energies in reference-index order (k corresponds to reference_states[k]).
    final_order : list[int]
        Permutation mapping: sorted index -> reference index.
        Example: final_order=[1,0] means the lowest energy came from reference 1.
    final_energies_sorted : list[float]
        Final energies sorted ascending (E0 <= E1 <= ...).
    """
    if not energies_per_state:
        return [], [], []

    finals_by_ref: list[float] = []
    for k, traj in enumerate(energies_per_state):
        if traj is None or len(traj) == 0:
            raise ValueError(f"energies_per_state[{k}] is empty; cannot sort finals.")
        finals_by_ref.append(float(traj[-1]))

    order = list(np.argsort(np.asarray(finals_by_ref, dtype=float)))
    finals_sorted = [finals_by_ref[i] for i in order]
    return finals_by_ref, order, finals_sorted


def _ensure_sorted_fields_in_result(result: dict) -> dict:
    """
    Ensure sorted-final fields exist in a result dict (for cache back-compat).

    This does NOT reorder trajectories; it only adds:
      - final_energies_by_ref
      - final_order
      - final_energies_sorted
    """
    if "energies_per_state" not in result:
        return result

    if (
        "final_energies_by_ref" in result
        and "final_order" in result
        and "final_energies_sorted" in result
    ):
        return result

    finals_by_ref, order, finals_sorted = _compute_sorted_finals(
        result["energies_per_state"]
    )
    result["final_energies_by_ref"] = finals_by_ref
    result["final_order"] = order
    result["final_energies_sorted"] = finals_sorted
    return result


def run_ssvqe(
    molecule: str = "H3+",
    *,
    num_states: int = 2,
    weights: Optional[Sequence[float]] = None,
    ansatz_name: str = "UCCSD",
    optimizer_name: str = "Adam",
    steps: int = 100,
    stepsize: float = 0.4,
    seed: int = 0,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[[list[int]], None]] = None,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    reference_states: Optional[Sequence[Sequence[int]]] = None,
    plot: bool = True,
    force: bool = False,
):
    """
    Subspace-Search VQE (SSVQE).

    Canonical structure:
        |psi_k(theta)> = U(theta) |phi_k>
        minimize sum_k w_k <psi_k(theta)| H |psi_k(theta)>

    Implementation notes (important)
    -------------------------------
    - This implementation is **ansatz-agnostic**:
        reference preparation |phi_k> is done explicitly via qml.BasisState.
      This ensures SSVQE works for toy ansatzes that do not support reference_state kwargs.

    - For UCC-family ansatzes, we prevent the ansatz from re-preparing HF by calling it with
      prepare_reference=False (if supported). This makes the ansatz act like a “pure” U(theta).

    Noise
    -----
    - Supports legacy depolarizing/amplitude knobs AND an arbitrary `noise_model(wires)` callable.
    - If `noisy=False`, no noise is applied even if noise_model is provided.

    Output ordering contract (important)
    ------------------------------------
    - energies_per_state[k] is ALWAYS tied to reference_states[k] (reference-index semantics).
    - To avoid “state swapping” in downstream comparisons, the result includes:
        * final_energies_by_ref   (reference-index finals)
        * final_order            (sorted index -> reference index)
        * final_energies_sorted  (ascending finals, canonical E0/E1/...)
      Trajectories are NOT reordered (energy curves can cross).
    """
    if num_states < 2:
        raise ValueError("SSVQE is typically used with num_states >= 2")

    np.random.seed(int(seed))
    ensure_dirs()

    # 1) Hamiltonian + molecular data
    if symbols is None or coordinates is None:
        H, num_wires, symbols, coordinates, basis_out = build_hamiltonian(molecule)
        basis = str(basis_out)
    else:
        # Best-effort: preserve old override behaviour
        # (charge inference kept minimal; preferred path is build_hamiltonian.)
        charge = +1 if str(molecule).strip().upper() == "H3+" else 0
        H, num_wires = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=charge, basis=str(basis), unit="angstrom"
        )

    # 2) Shared ansatz parameters
    ansatz_fn, p0 = build_ansatz(
        ansatz_name,
        num_wires,
        seed=int(seed),
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )
    params = np.array(p0, requires_grad=True)

    # 3) Reference states (orthogonal computational basis)
    if reference_states is None:
        if str(ansatz_name).strip().upper().startswith("UCC"):
            singles, doubles, hf_state = _build_ucc_data(
                symbols, coordinates, basis=basis
            )
            refs = _ucc_reference_states_from_excitations(
                hf_state,
                singles,
                doubles,
                num_states=int(num_states),
                include_doubles=True,
            )
            # If the generator returned fewer than requested, fill with default basis states.
            if len(refs) < int(num_states):
                fill = _default_reference_states(int(num_states), int(num_wires))
                for s in fill:
                    if len(refs) >= int(num_states):
                        break
                    if s not in refs:
                        refs.append(s)
            reference_states = refs[: int(num_states)]
        else:
            reference_states = _default_reference_states(
                int(num_states), int(num_wires)
            )

    reference_states = _validate_reference_states(
        reference_states,
        num_states=int(num_states),
        num_wires=int(num_wires),
    )

    # 4) Weights
    if weights is None:
        weights = [1.0 + float(k) for k in range(int(num_states))]
    weights = [float(w) for w in weights]
    if len(weights) != int(num_states):
        raise ValueError(f"weights must have length {num_states}, got {len(weights)}")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be strictly positive")

    # 5) Device + diff method
    dev = make_device(int(num_wires), noisy=bool(noisy))
    diff_method = "finite-diff" if noisy else "parameter-shift"

    # 6) Energy QNode: prepares |phi_k> explicitly, then applies U(theta)
    @qml.qnode(dev, diff_method=diff_method)
    def energy(theta, reference_state):
        # Always prepare the reference determinant explicitly (ansatz-agnostic)
        qml.BasisState(
            np.array(reference_state, dtype=int), wires=range(int(num_wires))
        )

        # Apply the ansatz as “U(theta)” only.
        # For chemistry ansatzes that support it, disable internal reference preparation.
        _call_ansatz(
            ansatz_fn,
            theta,
            wires=range(int(num_wires)),
            symbols=symbols,
            coordinates=coordinates,
            basis=basis,
            reference_state=None,
            prepare_reference=False,
        )

        apply_optional_noise(
            bool(noisy),
            float(depolarizing_prob),
            float(amplitude_damping_prob),
            int(num_wires),
            noise_model=noise_model,
        )
        return qml.expval(H)

    # 7) Config + caching
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=str(basis),
        ansatz_desc=f"SSVQE({ansatz_name})_{int(num_states)}states",
        optimizer_name=optimizer_name,
        stepsize=float(stepsize),
        max_iterations=int(steps),
        seed=int(seed),
        mapping="jordan_wigner",
        noisy=bool(noisy),
        depolarizing_prob=float(depolarizing_prob),
        amplitude_damping_prob=float(amplitude_damping_prob),
        molecule_label=molecule,
    )
    cfg["num_states"] = int(num_states)
    cfg["weights"] = [float(w) for w in weights]
    cfg["reference_states"] = [list(map(int, s)) for s in reference_states]
    cfg["noise_model"] = (
        getattr(noise_model, "__name__", str(noise_model))
        if noise_model is not None
        else None
    )

    sig = run_signature(cfg)
    prefix = make_filename_prefix(
        cfg, noisy=bool(noisy), seed=int(seed), hash_str=sig, algo="SSVQE"
    )
    result_path = RESULTS_DIR / f"{prefix}.json"

    if not force and result_path.exists():
        print(f"📂 Using cached SSVQE result: {result_path}")
        with result_path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        cached = record.get("result", {})
        return _ensure_sorted_fields_in_result(cached)

    # 8) Cost
    opt = build_optimizer(optimizer_name, stepsize=float(stepsize))
    energies_per_state: list[list[float]] = [[] for _ in range(int(num_states))]

    def cost(theta):
        total = 0.0
        for k in range(int(num_states)):
            total = total + weights[k] * energy(theta, reference_states[k])
        return total

    # 9) Optimization loop
    for _ in range(int(steps)):
        try:
            params, _ = opt.step_and_cost(cost, params)
        except AttributeError:
            params = opt.step(cost, params)

        # record each state's energy under current shared params (reference-index semantics)
        for k in range(int(num_states)):
            energies_per_state[k].append(float(energy(params, reference_states[k])))

    # 10) Final canonical (sorted) view
    finals_by_ref, final_order, finals_sorted = _compute_sorted_finals(
        energies_per_state
    )

    # 11) Save
    result = {
        # Reference-index trajectories (do NOT reorder)
        "energies_per_state": energies_per_state,
        "final_params": params.tolist(),
        "config": cfg,
        # Canonical finals (stable E0/E1/.. for comparisons)
        "final_energies_by_ref": finals_by_ref,
        "final_order": final_order,
        "final_energies_sorted": finals_sorted,
    }
    save_run_record(prefix, {"config": cfg, "result": result})
    print(f"💾 Saved SSVQE run to {result_path}")

    # 12) Optional plotting (E0/E1 only; plots reference-index trajectories)
    if plot and int(num_states) >= 2:
        try:
            plot_multi_state_convergence(
                ssvqe_or_vqd="SSVQE",
                molecule=molecule,
                ansatz=ansatz_name,
                optimizer_name=optimizer_name,
                E0_list=energies_per_state[0],
                E1_list=energies_per_state[1],
                show=True,
                save=True,
            )
        except Exception as exc:
            print(f"⚠️ SSVQE plotting failed (non-fatal): {exc}")

    return result
