# vqe/ssvqe.py
from __future__ import annotations

import json
from typing import List, Optional, Sequence

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
        raise ValueError(
            f"Invalid single excitation {exc}: orbital {i} not occupied in HF."
        )
    if det[a] != 0:
        raise ValueError(
            f"Invalid single excitation {exc}: orbital {a} already occupied in HF."
        )

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
        raise ValueError(
            f"Invalid double excitation {exc}: {i},{j} must be occupied in HF."
        )
    if det[a] != 0 or det[b] != 0:
        raise ValueError(
            f"Invalid double excitation {exc}: {a},{b} must be unoccupied in HF."
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

    # Singles first (usually the most relevant low-energy manifold)
    for exc in singles:
        if len(refs) >= num_states:
            return refs
        try:
            refs.append(_apply_single_excitation_to_det(hf, exc))
        except ValueError:
            # Skip invalid excitations for this HF (shouldn't happen, but safe)
            continue

    if include_doubles:
        for exc in doubles:
            if len(refs) >= num_states:
                return refs
            try:
                refs.append(_apply_double_excitation_to_det(hf, exc))
            except ValueError:
                continue

    # If still short, fall back to HF+bitflips to fill
    if len(refs) < num_states:
        n = len(hf)
        # single flips
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

    Note: for chemistry, you will usually want to pass your own states
    (e.g., HF plus low-rank excitations) rather than rely on this default.
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
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    reference_states: Optional[Sequence[Sequence[int]]] = None,
    plot: bool = True,
    force: bool = False,
):
    """
    True Subspace-Search VQE (SSVQE).

    Classic structure:
        |psi_k(theta)> = U(theta) |phi_k>
        minimize sum_k w_k <psi_k(theta)| H |psi_k(theta)>

    - A single shared parameter vector `theta` for all target states.
    - Orthogonality comes from distinct orthogonal inputs |phi_k> (no penalties).

    Parameters
    ----------
    num_states
        Number of eigenstates to target (>= 2 is typical).
    weights
        Positive weights w_k. Must be length `num_states`.
        If None, defaults to strictly increasing weights: w_k = 1 + k.
        (This biases the optimizer to align lower-energy states first.)
    reference_states
        Sequence of computational-basis bitstrings (length num_wires each),
        one per state. If None, uses a simple default basis set.

        For chemistry + UCC, passing meaningful reference determinants
        is strongly recommended (HF + excited determinants).
    """
    if num_states < 2:
        raise ValueError("SSVQE is typically used with num_states >= 2")

    np.random.seed(seed)
    ensure_dirs()

    # 1) Build Hamiltonian + molecular data
    if symbols is None or coordinates is None:
        H, num_wires, symbols, coordinates, basis = build_hamiltonian(molecule)
    else:
        charge = +1 if molecule.upper() == "H3+" else 0
        H, num_wires = qml.qchem.molecular_hamiltonian(
            symbols, coordinates, charge=charge, basis=basis, unit="angstrom"
        )

    # 2) Shared ansatz parameters
    ansatz_fn, p0 = build_ansatz(
        ansatz_name,
        num_wires,
        seed=seed,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
    )
    params = np.array(p0, requires_grad=True)

    # 3) Reference states
    if reference_states is None:
        if ansatz_name.upper().startswith("UCC"):
            # Use the exact excitation lists and HF state used by the UCC ansatz
            singles, doubles, hf_state = _build_ucc_data(
                symbols, coordinates, basis=basis
            )
            reference_states = _ucc_reference_states_from_excitations(
                hf_state,
                singles,
                doubles,
                num_states=num_states,
                include_doubles=True,
            )
        else:
            reference_states = _default_reference_states(num_states, num_wires)
    else:
        reference_states = [list(s) for s in reference_states]

    # Ensure all reference states are distinct (orthogonal determinants)
    if len({tuple(s) for s in reference_states}) != len(reference_states):
        raise ValueError(
            "reference_states contain duplicates; cannot enforce orthogonality."
        )

    # 4) Weights
    if weights is None:
        weights = [1.0 + float(k) for k in range(num_states)]
    weights = [float(w) for w in weights]
    if len(weights) != num_states:
        raise ValueError(f"weights must have length {num_states}, got {len(weights)}")
    if any(w <= 0 for w in weights):
        raise ValueError("weights must be strictly positive")

    # 5) Device
    dev = make_device(num_wires, noisy=noisy)

    diff_method = "finite-diff" if noisy else "parameter-shift"

    # 6) Energy QNode that accepts a reference_state selector
    @qml.qnode(dev, diff_method=diff_method)
    def energy(params, reference_state):
        _call_ansatz(
            ansatz_fn,
            params,
            wires=range(num_wires),
            symbols=symbols,
            coordinates=coordinates,
            basis=basis,
            reference_state=reference_state,
            prepare_reference=True,
        )
        apply_optional_noise(
            noisy,
            depolarizing_prob,
            amplitude_damping_prob,
            num_wires,
        )
        return qml.expval(H)

    # 7) Config + caching
    cfg = make_run_config_dict(
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        ansatz_desc=f"SSVQE_TRUE({ansatz_name})_{num_states}states",
        optimizer_name=optimizer_name,
        stepsize=stepsize,
        max_iterations=steps,
        seed=seed,
        mapping="jordan_wigner",
        noisy=noisy,
        depolarizing_prob=depolarizing_prob,
        amplitude_damping_prob=amplitude_damping_prob,
        molecule_label=molecule,
    )
    cfg["num_states"] = int(num_states)
    cfg["weights"] = [float(w) for w in weights]
    cfg["reference_states"] = [list(map(int, s)) for s in reference_states]

    sig = run_signature(cfg)
    prefix = make_filename_prefix(cfg, noisy=noisy, seed=seed, hash_str=sig, algo="SSVQE")
    result_path = RESULTS_DIR / f"{prefix}.json"

    if not force and result_path.exists():
        print(f"📂 Using cached SSVQE result: {result_path}")
        with open(result_path, "r") as f:
            record = json.load(f)
        return record["result"]

    # 8) Cost: weighted sum of per-state energies
    opt = build_optimizer(optimizer_name, stepsize=stepsize)

    energies_per_state = [[] for _ in range(num_states)]

    def cost(theta):
        total = 0.0
        for k in range(num_states):
            total = total + weights[k] * energy(theta, reference_states[k])
        return total

    # 9) Optimization loop
    for _ in range(steps):
        try:
            params, _ = opt.step_and_cost(cost, params)
        except AttributeError:
            params = opt.step(cost, params)

        # record each state's energy under current shared params
        for k in range(num_states):
            energies_per_state[k].append(float(energy(params, reference_states[k])))

    # 10) Save
    result = {
        "energies_per_state": energies_per_state,
        "final_params": params.tolist(),
        "config": cfg,
    }
    save_run_record(prefix, {"config": cfg, "result": result})
    print(f"💾 Saved SSVQE run to {result_path}")

    # 11) Optional plotting (E0/E1 only)
    if plot and num_states >= 2:
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
