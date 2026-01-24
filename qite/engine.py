"""
qite.engine
===========
Circuit/QNode plumbing for imaginary-time / QITE-style workflows.

Public surface:
- make_device(...)
- make_energy_qnode(...)
- make_state_qnode(...)
- build_ansatz(...)
- qite_step(...)   <-- VarQITE (McLachlan) parameter update

VarQITE update rule
-------------------
We implement a "true" variational imaginary-time evolution step via
McLachlan's variational principle:

    A(θ) θ_dot = -C(θ)

with
    A_ij = Re( <∂_i ψ(θ) | ∂_j ψ(θ)> )
    C_i  = Re( <∂_i ψ(θ) | (H - E(θ)) | ψ(θ)> )
with tangent-space (Fubini–Study) projection applied to ∂_i ψ.

and update:
    θ <- θ + dtau * θ_dot

Notes
-----
- This step requires a *pure state* |ψ(θ)⟩ (statevector). Mixed-state noise
  evolution is intentionally not supported for "true VarQITE".
- We use central finite differences to approximate |∂_i ψ⟩. This is slower
  than analytic derivatives but robust and keeps the code minimal.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import pennylane as qml
from pennylane import numpy as np


# ================================================================
# DEVICE
# ================================================================
def make_device(
    num_wires: int,
    *,
    noisy: bool = False,
    shots: Optional[int] = None,
):
    """
    Create a PennyLane device consistent with the VQE/QPE conventions.

    - noiseless: default.qubit (statevector)
    - noisy:     default.mixed (density matrix)

    shots=None keeps analytic mode (recommended for these demos).
    """
    dev_name = "default.mixed" if bool(noisy) else "default.qubit"
    return qml.device(dev_name, wires=int(num_wires), shots=shots)


# ================================================================
# NOISE APPLICATION (VQE-like)
# ================================================================
def _apply_noise_layer(
    *,
    num_wires: int,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[..., Any]] = None,
):
    """
    Apply a noise layer after an ansatz.

    Priority:
      1) user-provided noise_model(...) if given
      2) built-in depolarizing / amplitude damping (best-effort)
    """
    p_dep = float(depolarizing_prob)
    p_amp = float(amplitude_damping_prob)

    if noise_model is not None:
        noise_model(
            num_wires=int(num_wires),
            depolarizing_prob=p_dep,
            amplitude_damping_prob=p_amp,
        )
        return

    if p_dep > 0.0:
        for w in range(int(num_wires)):
            qml.DepolarizingChannel(p_dep, wires=w)

    if p_amp > 0.0:
        for w in range(int(num_wires)):
            qml.AmplitudeDamping(p_amp, wires=w)


# ================================================================
# ANSATZ BUILDING
# ================================================================
def _fallback_hardware_efficient_ansatz(
    params: np.ndarray,
    *,
    num_wires: int,
    layers: int,
):
    """
    Minimal hardware-efficient ansatz:
      - per-layer: RY then RZ on each wire
      - entangling: CNOT chain
    """
    for layer in range(int(layers)):
        for w in range(int(num_wires)):
            qml.RY(params[layer, w, 0], wires=w)
            qml.RZ(params[layer, w, 1], wires=w)

        for w in range(int(num_wires) - 1):
            qml.CNOT(wires=[w, w + 1])


def build_ansatz(
    ansatz_name: str,
    num_wires: int,
    *,
    seed: int = 0,
    symbols=None,
    coordinates=None,
    basis: Optional[str] = None,
    requires_grad: bool = True,
    hf_state: Optional[np.ndarray] = None,
) -> Tuple[Callable[[np.ndarray], None], np.ndarray]:
    name = str(ansatz_name).strip()
    np.random.seed(int(seed))

    hf = None if hf_state is None else np.array(hf_state, dtype=int)

    try:
        import vqe.ansatz as _vqe_ansatz_mod  # type: ignore

        if hasattr(_vqe_ansatz_mod, "build_ansatz") and callable(
            _vqe_ansatz_mod.build_ansatz
        ):
            inner_ans_fn, init = _vqe_ansatz_mod.build_ansatz(
                name,
                int(num_wires),
                seed=int(seed),
                symbols=symbols,
                coordinates=coordinates,
                basis=basis,
                requires_grad=bool(requires_grad),
            )
            init_params = np.array(init, requires_grad=bool(requires_grad))

            def ansatz_fn(params):
                if hf is not None:
                    qml.BasisState(hf, wires=list(range(int(num_wires))))
                inner_ans_fn(params)

            return ansatz_fn, init_params
    except Exception:
        pass

    layers = 2
    init_params = 0.01 * np.random.randn(int(layers), int(num_wires), 2)
    init_params = np.array(init_params, requires_grad=bool(requires_grad))

    def ansatz_fn(params):
        if hf is not None:
            qml.BasisState(hf, wires=list(range(int(num_wires))))
        _fallback_hardware_efficient_ansatz(
            params, num_wires=int(num_wires), layers=int(layers)
        )

    return ansatz_fn, init_params


# ================================================================
# QNODES
# ================================================================
def make_energy_qnode(
    H: qml.Hamiltonian,
    dev,
    ansatz_fn: Callable[[np.ndarray], None],
    num_wires: int,
    *,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[..., Any]] = None,
    symbols=None,
    coordinates=None,
    basis: Optional[str] = None,
):
    """
    Construct an energy QNode E(θ) = <ψ(θ)|H|ψ(θ)> (or Tr[ρ(θ) H] for noisy).
    """

    @qml.qnode(dev, diff_method="parameter-shift")
    def energy_qnode(params):
        ansatz_fn(params)
        if bool(noisy):
            _apply_noise_layer(
                num_wires=int(num_wires),
                depolarizing_prob=float(depolarizing_prob),
                amplitude_damping_prob=float(amplitude_damping_prob),
                noise_model=noise_model,
            )
        return qml.expval(H)

    return energy_qnode


def make_state_qnode(
    dev,
    ansatz_fn: Callable[[np.ndarray], None],
    num_wires: int,
    *,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[..., Any]] = None,
    symbols=None,
    coordinates=None,
    basis: Optional[str] = None,
):
    if bool(noisy):

        @qml.qnode(dev)
        def state_qnode(params):
            ansatz_fn(params)
            _apply_noise_layer(
                num_wires=int(num_wires),
                depolarizing_prob=float(depolarizing_prob),
                amplitude_damping_prob=float(amplitude_damping_prob),
                noise_model=noise_model,
            )
            return qml.density_matrix(wires=list(range(int(num_wires))))

        return state_qnode

    @qml.qnode(dev)
    def state_qnode(params):
        ansatz_fn(params)
        return qml.state()

    return state_qnode


# ================================================================
# VARQITE (McLachlan) STEP
# ================================================================
def _flatten_params(params: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.array(params)
    shape = tuple(arr.shape)
    flat = np.reshape(arr, (-1,))
    return flat, shape


def _unflatten_params(flat: np.ndarray, shape: tuple[int, ...], *, requires_grad: bool):
    out = np.reshape(np.array(flat), shape)
    return np.array(out, requires_grad=bool(requires_grad))


def _ensure_statevector(state) -> np.ndarray:
    s = np.array(state)
    if s.ndim != 1:
        raise ValueError(
            "VarQITE requires a pure statevector |ψ(θ)⟩. "
            "Got a non-1D object (likely a density matrix). "
            "Run VarQITE in noiseless mode (default.qubit)."
        )
    # Defensive normalization (qml.state is already normalized, but be safe).
    nrm = np.linalg.norm(s)
    if float(nrm) == 0.0:
        raise ValueError("Statevector has zero norm (unexpected).")
    return s / nrm


def _hamiltonian_action(
    op,
    psi,
    *,
    num_wires: int,
    cache: Optional[dict] = None,
):
    cache = cache if cache is not None else {}

    psi = np.asarray(psi)
    if psi.ndim != 1:
        raise ValueError("Expected a statevector (1D).")

    key = "H_matrix"
    Hmat = cache.get(key, None)
    if Hmat is None:
        Hmat = qml.matrix(op, wire_order=list(range(int(num_wires))))
        Hmat = np.asarray(Hmat, dtype=complex)
        cache[key] = Hmat

    return Hmat @ psi


def _project_tangent(psi: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    psi = np.asarray(psi)
    dpsi = np.asarray(dpsi)
    overlaps = np.vdot(psi, dpsi)  # <psi|dpsi>
    return dpsi - overlaps * psi


def _finite_difference_state_derivatives(
    state_qnode: Callable[[np.ndarray], Any],
    params: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Central finite differences for |∂_i ψ(θ)⟩.

    Returns
    -------
    psi0 : (D,) complex
    dpsi : (P, D) complex
        P = number of parameters (flattened), D = state dimension.
    """
    flat, shape = _flatten_params(params)
    P = int(flat.shape[0])

    psi0 = _ensure_statevector(state_qnode(params))

    dpsis = []
    for i in range(P):
        fp = np.array(flat)
        fm = np.array(flat)
        fp[i] = fp[i] + float(eps)
        fm[i] = fm[i] - float(eps)

        params_p = _unflatten_params(fp, shape, requires_grad=False)
        params_m = _unflatten_params(fm, shape, requires_grad=False)

        psi_p = _ensure_statevector(state_qnode(params_p))
        psi_m = _ensure_statevector(state_qnode(params_m))

        dpsi = (psi_p - psi_m) / (2.0 * float(eps))
        dpsis.append(np.array(dpsi))

    dpsi_mat = np.stack(dpsis, axis=0) if P > 0 else np.zeros((0, psi0.size))
    return np.array(psi0), np.array(dpsi_mat)


def qite_step(
    *,
    params: np.ndarray,
    energy_qnode: Callable[[np.ndarray], Any],
    state_qnode: Callable[[np.ndarray], Any],
    dtau: float,
    num_wires: int,
    symbols=None,
    coordinates=None,
    basis: Optional[str] = None,
    hamiltonian: Optional[qml.Hamiltonian] = None,
    fd_eps: float = 1e-3,
    reg: float = 1e-6,
    solver: str = "solve",
    pinv_rcond: float = 1e-10,
    cache: Optional[dict] = None,
) -> np.ndarray:
    if hamiltonian is None:
        raise ValueError("qite_step requires `hamiltonian`.")

    psi, dpsi = _finite_difference_state_derivatives(
        state_qnode, params, eps=float(fd_eps)
    )

    P = int(dpsi.shape[0])
    if P == 0:
        return np.array(params, requires_grad=True)

    E = float(energy_qnode(params))
    Hpsi = _hamiltonian_action(hamiltonian, psi, num_wires=int(num_wires), cache=cache)

    dpsi_t = np.stack([_project_tangent(psi, dpsi[i]) for i in range(P)], axis=0)

    A = np.zeros((P, P), dtype=float)
    C = np.zeros((P,), dtype=float)

    for i in range(P):
        dpi = dpsi_t[i]
        C[i] = float(np.real(np.vdot(dpi, Hpsi) - E * np.vdot(dpi, psi)))
        for j in range(i, P):
            val = float(np.real(np.vdot(dpi, dpsi_t[j])))
            A[i, j] = val
            A[j, i] = val

    A = A + float(reg) * np.eye(P, dtype=float)

    b = -C
    solver_l = str(solver).strip().lower()

    try:
        if solver_l == "solve":
            v = np.linalg.solve(A, b)
        elif solver_l == "lstsq":
            v, *_ = np.linalg.lstsq(A, b, rcond=None)
        elif solver_l == "pinv":
            v = np.linalg.pinv(A, rcond=float(pinv_rcond)) @ b
        else:
            raise ValueError("solver must be one of: 'solve', 'lstsq', 'pinv'.")
    except Exception:
        try:
            v, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            v = np.linalg.pinv(A, rcond=float(pinv_rcond)) @ b

    v = np.array(v, dtype=float)

    flat, shape = _flatten_params(params)
    new_flat = np.array(flat, dtype=float) + float(dtau) * v
    new_params = _unflatten_params(new_flat, shape, requires_grad=True)

    return new_params
