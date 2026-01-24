"""
qite.engine
===========
Circuit/QNode plumbing for VarQITE (McLachlan) workflows.

Public surface
--------------
- make_device(...)
- build_ansatz(...)
- make_energy_qnode(...)
- make_state_qnode(...)
- qite_step(...)

Notes
-----
- VarQITE updates require a pure statevector |ψ(θ)⟩ (default.qubit).
- Noise support here is only for post-evaluation (default.mixed) QNodes.
- State derivatives are computed via central finite differences for simplicity.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import pennylane as qml
from pennylane import numpy as np


# =============================================================================
# DEVICE
# =============================================================================
def make_device(
    num_wires: int,
    *,
    noisy: bool = False,
    shots: Optional[int] = None,
):
    """
    Create a PennyLane device.

    noiseless -> default.qubit (statevector)
    noisy     -> default.mixed (density matrix)

    shots=None keeps analytic mode.
    """
    dev_name = "default.mixed" if bool(noisy) else "default.qubit"
    return qml.device(dev_name, wires=int(num_wires), shots=shots)


# =============================================================================
# NOISE LAYER (post-evaluation only)
# =============================================================================
def _apply_noise_layer(
    *,
    num_wires: int,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[..., Any]] = None,
) -> None:
    """
    Apply noise after an ansatz.

    Priority:
      1) noise_model(...) if provided
      2) built-in depolarizing / amplitude damping channels

    This is intended for post-evaluation QNodes (default.mixed), not VarQITE updates.
    """
    p_dep = float(depolarizing_prob)
    p_amp = float(amplitude_damping_prob)

    if (p_dep <= 0.0) and (p_amp <= 0.0) and (noise_model is None):
        return

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


# =============================================================================
# ANSATZ BUILDING
# =============================================================================
def _fallback_hardware_efficient_ansatz(
    params: np.ndarray,
    *,
    num_wires: int,
    layers: int,
) -> None:
    """
    Minimal hardware-efficient ansatz:
      - per-layer: RY then RZ on each wire
      - entangling: CNOT chain
    params shape: (layers, num_wires, 2)
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
    """
    Build an ansatz callable and an initial parameter array.

    If vqe.ansatz.build_ansatz is available, we delegate to it (to keep ansatz
    definitions centralized). Otherwise we fall back to a small HEA.

    The returned ansatz_fn(params) is responsible for preparing |ψ(θ)⟩, including
    optional HF initialization via qml.BasisState(hf_state, ...).
    """
    name = str(ansatz_name).strip()
    n = int(num_wires)
    np.random.seed(int(seed))

    hf = None if hf_state is None else np.array(hf_state, dtype=int)

    # Preferred: reuse VQE ansatz library if present.
    try:
        import vqe.ansatz as _vqe_ansatz_mod  # type: ignore

        builder = getattr(_vqe_ansatz_mod, "build_ansatz", None)
        if callable(builder):
            inner_fn, init = builder(
                name,
                n,
                seed=int(seed),
                symbols=symbols,
                coordinates=coordinates,
                basis=basis,
                requires_grad=bool(requires_grad),
            )
            init_params = np.array(init, requires_grad=bool(requires_grad))

            def ansatz_fn(params: np.ndarray) -> None:
                if hf is not None:
                    qml.BasisState(hf, wires=list(range(n)))
                inner_fn(params)

            return ansatz_fn, init_params
    except Exception:
        # Fall back below.
        pass

    # Fallback HEA
    layers = 2
    init = 0.01 * np.random.randn(int(layers), n, 2)
    init_params = np.array(init, requires_grad=bool(requires_grad))

    def ansatz_fn(params: np.ndarray) -> None:
        if hf is not None:
            qml.BasisState(hf, wires=list(range(n)))
        _fallback_hardware_efficient_ansatz(params, num_wires=n, layers=int(layers))

    return ansatz_fn, init_params


# =============================================================================
# QNODES
# =============================================================================
def make_energy_qnode(
    H, dev, ansatz_fn, num_wires, *,
    noisy=False,
    depolarizing_prob=0.0,
    amplitude_damping_prob=0.0,
    noise_model=None,
    **_ignored,
):
    """
    Energy QNode:
      - noiseless:  ⟨ψ(θ)|H|ψ(θ)⟩
      - noisy:      Tr[ρ(θ) H]  (noise applied after ansatz)
    """
    n = int(num_wires)

    @qml.qnode(dev)
    def energy_qnode(params: np.ndarray):
        ansatz_fn(params)
        if bool(noisy):
            _apply_noise_layer(
                num_wires=n,
                depolarizing_prob=float(depolarizing_prob),
                amplitude_damping_prob=float(amplitude_damping_prob),
                noise_model=noise_model,
            )
        return qml.expval(H)

    return energy_qnode


def make_state_qnode(
    dev, ansatz_fn, num_wires, *,
    noisy=False,
    depolarizing_prob=0.0,
    amplitude_damping_prob=0.0,
    noise_model=None,
    **_ignored,
):
    """
    State QNode:
      - noiseless: returns qml.state() (statevector)
      - noisy: returns qml.density_matrix(...)
    """
    n = int(num_wires)

    if bool(noisy):

        @qml.qnode(dev)
        def state_qnode(params: np.ndarray):
            ansatz_fn(params)
            _apply_noise_layer(
                num_wires=n,
                depolarizing_prob=float(depolarizing_prob),
                amplitude_damping_prob=float(amplitude_damping_prob),
                noise_model=noise_model,
            )
            return qml.density_matrix(wires=list(range(n)))

        return state_qnode

    @qml.qnode(dev)
    def state_qnode(params: np.ndarray):
        ansatz_fn(params)
        return qml.state()

    return state_qnode


# =============================================================================
# VARQITE (McLachlan) STEP
# =============================================================================
def _flatten_params(params: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    arr = np.array(params)
    shape = tuple(arr.shape)
    flat = np.reshape(arr, (-1,))
    return flat, shape


def _unflatten_params(
    flat: np.ndarray,
    shape: Tuple[int, ...],
    *,
    requires_grad: bool,
) -> np.ndarray:
    out = np.reshape(np.array(flat), shape)
    return np.array(out, requires_grad=bool(requires_grad))


def _ensure_statevector(state) -> np.ndarray:
    s = np.array(state)
    if s.ndim != 1:
        raise ValueError(
            "VarQITE requires a pure statevector |ψ(θ)⟩ (default.qubit). "
            "Got a non-1D object (likely a density matrix)."
        )
    nrm = np.linalg.norm(s)
    if float(nrm) == 0.0:
        raise ValueError("Statevector has zero norm.")
    return s / nrm


def _project_tangent(psi: np.ndarray, dpsi: np.ndarray) -> np.ndarray:
    overlap = np.vdot(psi, dpsi)  # <psi|dpsi>
    return dpsi - overlap * psi


def _hamiltonian_action(
    H: qml.Hamiltonian,
    psi: np.ndarray,
    *,
    num_wires: int,
    cache: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute H|ψ⟩ by materializing the Hamiltonian matrix once (cached) and multiplying.
    """
    cache = {} if cache is None else cache
    key = "H_matrix"

    Hmat = cache.get(key)
    if Hmat is None:
        Hmat = qml.matrix(H, wire_order=list(range(int(num_wires))))
        Hmat = np.asarray(Hmat, dtype=complex)
        cache[key] = Hmat

    return np.asarray(Hmat @ np.asarray(psi), dtype=complex)


def _finite_difference_state_derivatives(
    state_qnode: Callable[[np.ndarray], Any],
    params: np.ndarray,
    *,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central finite differences for |∂_i ψ(θ)⟩.

    Returns
    -------
    psi0 : (D,) complex
    dpsi : (P, D) complex
    """
    flat, shape = _flatten_params(params)
    P = int(flat.shape[0])

    psi0 = _ensure_statevector(state_qnode(params))

    if P == 0:
        return np.array(psi0), np.zeros((0, psi0.size), dtype=complex)

    dpsis = []
    eps_f = float(eps)

    for i in range(P):
        fp = np.array(flat, dtype=float)
        fm = np.array(flat, dtype=float)
        fp[i] = fp[i] + eps_f
        fm[i] = fm[i] - eps_f

        params_p = _unflatten_params(fp, shape, requires_grad=False)
        params_m = _unflatten_params(fm, shape, requires_grad=False)

        psi_p = _ensure_statevector(state_qnode(params_p))
        psi_m = _ensure_statevector(state_qnode(params_m))

        dpsi = (psi_p - psi_m) / (2.0 * eps_f)
        dpsis.append(np.array(dpsi, dtype=complex))

    dpsi_mat = np.stack(dpsis, axis=0)
    return np.array(psi0, dtype=complex), np.array(dpsi_mat, dtype=complex)


def qite_step(
    *,
    params: np.ndarray,
    energy_qnode: Callable[[np.ndarray], Any],
    state_qnode: Callable[[np.ndarray], Any],
    dtau: float,
    num_wires: int,
    hamiltonian: qml.Hamiltonian,
    fd_eps: float = 1e-3,
    reg: float = 1e-6,
    solver: str = "solve",
    pinv_rcond: float = 1e-10,
    cache: Optional[dict] = None,
) -> np.ndarray:
    """
    One McLachlan VarQITE update step:

        A(θ) v = -C(θ)
        θ <- θ + dtau * v

    where A_ij = Re(<∂i ψ|∂j ψ>) and C_i = Re(<∂i ψ|(H-E)|ψ>),
    with tangent-space projection applied to the derivatives.
    """
    if hamiltonian is None:
        raise ValueError("qite_step requires `hamiltonian`.")

    psi, dpsi = _finite_difference_state_derivatives(
        state_qnode, params, eps=float(fd_eps)
    )
    P = int(dpsi.shape[0])
    if P == 0:
        return np.array(params, requires_grad=True)

    # Tangent-space projection
    dpsi_t = np.stack([_project_tangent(psi, dpsi[i]) for i in range(P)], axis=0)

    E = float(energy_qnode(params))
    Hpsi = _hamiltonian_action(hamiltonian, psi, num_wires=int(num_wires), cache=cache)

    A = np.zeros((P, P), dtype=float)
    C = np.zeros((P,), dtype=float)

    for i in range(P):
        dpi = dpsi_t[i]
        # C_i = Re(<∂iψ|H|ψ> - E <∂iψ|ψ>)
        C[i] = float(np.real(np.vdot(dpi, Hpsi) - E * np.vdot(dpi, psi)))

        for j in range(i, P):
            val = float(np.real(np.vdot(dpi, dpsi_t[j])))
            A[i, j] = val
            A[j, i] = val

    # Tikhonov regularization
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
        # Robust fallback cascade
        try:
            v, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            v = np.linalg.pinv(A, rcond=float(pinv_rcond)) @ b

    flat, shape = _flatten_params(params)
    new_flat = np.array(flat, dtype=float) + float(dtau) * np.array(v, dtype=float)
    return _unflatten_params(new_flat, shape, requires_grad=True)
