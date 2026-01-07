"""
vqe.engine
----------
Core plumbing layer for VQE / SSVQE / VQD routines.

Responsibilities
----------------
- Device creation and noise insertion
- Ansatz construction and parameter initialisation
- Optimizer creation
- QNode builders for:
    * energy expectation values
    * final states (statevector or density matrix)
    * overlap/fidelity-style quantities

Noise design (updated)
----------------------
This module supports two noise interfaces:

1) Legacy convenience parameters (backwards compatible):
      noisy=True/False
      depolarizing_prob=...
      amplitude_damping_prob=...

2) General / extensible noise model:
      noise_model: Optional[Callable[[list[int]], None]]

   where `noise_model(wires)` applies PennyLane channels to those wires.

If `noise_model` is provided, it is applied in addition to (or instead of)
legacy dep/amp settings. This enables "any noise type" without modifying
algorithms like VQD.
"""

from __future__ import annotations

import inspect
from typing import Callable, Iterable, Optional

import pennylane as qml

from .ansatz import get_ansatz, init_params
from .optimizer import get_optimizer


# ======================================================================
# DEVICE & NOISE HANDLING
# ======================================================================
def make_device(num_wires: int, noisy: bool = False):
    """
    Construct a PennyLane device.

    Parameters
    ----------
    num_wires
        Number of qubits.
    noisy
        If True, use a mixed-state simulator (`default.mixed`);
        otherwise use a statevector simulator (`default.qubit`).
    """
    dev_name = "default.mixed" if noisy else "default.qubit"
    return qml.device(dev_name, wires=num_wires)


def _apply_legacy_noise_channels(
    depolarizing_prob: float,
    amplitude_damping_prob: float,
    wires: list[int],
):
    """
    Apply the legacy built-in noise channels, if probabilities are > 0.
    """
    for w in wires:
        if depolarizing_prob and depolarizing_prob > 0.0:
            qml.DepolarizingChannel(float(depolarizing_prob), wires=w)
        if amplitude_damping_prob and amplitude_damping_prob > 0.0:
            qml.AmplitudeDamping(float(amplitude_damping_prob), wires=w)


def apply_optional_noise(
    noisy: bool,
    depolarizing_prob: float,
    amplitude_damping_prob: float,
    num_wires: int,
    *,
    noise_model: Optional[Callable[[list[int]], None]] = None,
):
    """
    Apply optional noise channels after the ansatz.

    Intended to be called from inside a QNode *after* the variational circuit.

    Parameters
    ----------
    noisy
        Whether noise is enabled.
    depolarizing_prob
        Legacy depolarizing probability (applied if > 0).
    amplitude_damping_prob
        Legacy amplitude damping probability (applied if > 0).
    num_wires
        Number of qubits.
    noise_model
        Optional callable noise model: noise_model(wires) -> None.
        If provided, it will be applied when `noisy=True`.

    Notes
    -----
    - If `noisy=False`, this function is a no-op, even if noise_model is provided.
    - If `noisy=True`, this applies both:
        (a) legacy dep/amp (if nonzero), and
        (b) noise_model (if provided).
      This keeps backwards compatibility while enabling extensibility.
    """
    if not noisy:
        return

    wires = list(range(int(num_wires)))

    # Legacy built-ins
    _apply_legacy_noise_channels(depolarizing_prob, amplitude_damping_prob, wires)

    # Extensible user-provided noise model
    if noise_model is not None:
        noise_model(wires)


# ======================================================================
# ANSATZ CONSTRUCTION
# ======================================================================

_ANSATZ_KWARG_CACHE: dict[Callable, set[str]] = {}


def _supported_ansatz_kwargs(ansatz_fn: Callable) -> set[str]:
    """Return the set of supported keyword argument names for an ansatz."""
    if ansatz_fn in _ANSATZ_KWARG_CACHE:
        return _ANSATZ_KWARG_CACHE[ansatz_fn]

    sig = inspect.signature(ansatz_fn).parameters
    supported = {
        name
        for name, p in sig.items()
        if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
    }
    _ANSATZ_KWARG_CACHE[ansatz_fn] = supported
    return supported


def _call_ansatz(
    ansatz_fn: Callable,
    params,
    wires: Iterable[int],
    symbols=None,
    coordinates=None,
    reference_state=None,
    prepare_reference: Optional[bool] = None,
    basis: Optional[str] = None,
):
    """
    Call an ansatz function, forwarding only the keyword arguments it supports.

    This unifies toy ansatzes (expecting (params, wires)) and chemistry
    ansatzes (which additionally accept symbols / coordinates / basis).

    Extra kwargs supported by chemistry ansatzes:
      - symbols, coordinates, basis
      - reference_state, prepare_reference
    """
    wires = list(wires)
    supported = _supported_ansatz_kwargs(ansatz_fn)

    kwargs = {}
    if "symbols" in supported:
        kwargs["symbols"] = symbols
    if "coordinates" in supported:
        kwargs["coordinates"] = coordinates
    if "basis" in supported and basis is not None:
        kwargs["basis"] = basis
    if "reference_state" in supported:
        kwargs["reference_state"] = reference_state
    if "prepare_reference" in supported and prepare_reference is not None:
        kwargs["prepare_reference"] = prepare_reference

    return ansatz_fn(params, wires=wires, **kwargs)


def build_ansatz(
    ansatz_name: str,
    num_wires: int,
    *,
    seed: int = 0,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    requires_grad: bool = True,
    scale: float = 0.01,
):
    """
    Construct an ansatz function and matching initial parameter vector.

    Returns
    -------
    (ansatz_fn, params)
        ansatz_fn: Callable(params, wires=...) -> circuit
        params:    numpy array of initial parameters
    """
    ansatz_fn = get_ansatz(ansatz_name)
    params = init_params(
        ansatz_name=ansatz_name,
        num_wires=num_wires,
        scale=scale,
        requires_grad=requires_grad,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        seed=seed,
    )
    return ansatz_fn, params


# ======================================================================
# OPTIMIZER BUILDER
# ======================================================================
def build_optimizer(optimizer_name: str, stepsize: float):
    """
    Return a PennyLane optimizer instance by name.
    """
    return get_optimizer(optimizer_name, stepsize=stepsize)


# ======================================================================
# QNODE CONSTRUCTION
# ======================================================================
def _choose_diff_method(noisy: bool, diff_method: Optional[str]) -> str:
    """
    Decide which differentiation method to use for a QNode.

    Default:
        - parameter-shift  when noiseless
        - finite-diff      when noisy
    """
    if diff_method is not None:
        return diff_method
    return "finite-diff" if noisy else "parameter-shift"


def make_energy_qnode(
    H,
    dev,
    ansatz_fn: Callable,
    num_wires: int,
    *,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[[list[int]], None]] = None,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    diff_method: Optional[str] = None,
):
    """
    Build a QNode that returns the energy expectation value ⟨H⟩.

    Returns
    -------
    energy(params) -> float
    """
    diff_method = _choose_diff_method(noisy, diff_method)

    @qml.qnode(dev, diff_method=diff_method)
    def energy(params):
        _call_ansatz(
            ansatz_fn,
            params,
            wires=range(num_wires),
            symbols=symbols,
            coordinates=coordinates,
            basis=basis,
        )
        apply_optional_noise(
            noisy,
            depolarizing_prob,
            amplitude_damping_prob,
            num_wires,
            noise_model=noise_model,
        )
        return qml.expval(H)

    return energy


def make_state_qnode(
    dev,
    ansatz_fn: Callable,
    num_wires: int,
    *,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[[list[int]], None]] = None,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    diff_method: Optional[str] = None,
):
    """
    Build a QNode that returns the final state for given parameters.

    For noiseless devices (default.qubit) this returns a statevector.
    For mixed-state devices (default.mixed) this returns a density matrix.

    Returns
    -------
    state(params) -> np.ndarray
    """
    diff_method = _choose_diff_method(noisy, diff_method)

    @qml.qnode(dev, diff_method=diff_method)
    def state(params):
        _call_ansatz(
            ansatz_fn,
            params,
            wires=range(num_wires),
            symbols=symbols,
            coordinates=coordinates,
            basis=basis,
        )
        apply_optional_noise(
            noisy,
            depolarizing_prob,
            amplitude_damping_prob,
            num_wires,
            noise_model=noise_model,
        )
        return qml.state()

    return state


def make_overlap00_fn(
    dev,
    ansatz_fn: Callable,
    num_wires: int,
    *,
    noisy: bool = False,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    noise_model: Optional[Callable[[list[int]], None]] = None,
    symbols=None,
    coordinates=None,
    basis: str = "sto-3g",
    diff_method: Optional[str] = None,
):
    """
    Construct a function overlap00(p_i, p_j) ≈ |⟨ψ_i|ψ_j⟩|².

    Uses the "adjoint trick":
        1) Prepare |ψ_i⟩ with ansatz(params=p_i)
        2) Apply adjoint(ansatz)(params=p_j)
        3) Measure probabilities; |⟨ψ_i|ψ_j⟩|² = Prob(|00...0⟩)

    Returns
    -------
    overlap00(p_i, p_j) -> float
    """
    diff_method = _choose_diff_method(noisy, diff_method)

    def _apply(params):
        _call_ansatz(
            ansatz_fn,
            params,
            wires=range(num_wires),
            symbols=symbols,
            coordinates=coordinates,
            basis=basis,
        )
        apply_optional_noise(
            noisy,
            depolarizing_prob,
            amplitude_damping_prob,
            num_wires,
            noise_model=noise_model,
        )

    @qml.qnode(dev, diff_method=diff_method)
    def _overlap(p_i, p_j):
        _apply(p_i)
        qml.adjoint(_apply)(p_j)
        return qml.probs(wires=range(num_wires))

    def overlap00(p_i, p_j):
        probs = _overlap(p_i, p_j)
        return probs[0]

    return overlap00
