"""
vqe.engine
----------
Core plumbing layer for VQE and SSVQE routines.

Handles:
- Device creation and optional noise insertion
- Ansatz construction and initialization
- Optimizer creation
- QNode builders for energy and overlap evaluation
"""

from __future__ import annotations
import inspect
import pennylane as qml
from pennylane import numpy as np

from .ansatz import get_ansatz, init_params
from .optimizer import get_optimizer


# ================================================================
# DEVICE & NOISE HANDLING
# ================================================================
def make_device(num_wires: int, noisy: bool = False):
    """Return a PennyLane device, using a mixed-state simulator if noise is enabled."""
    dev_name = "default.mixed" if noisy else "default.qubit"
    return qml.device(dev_name, wires=num_wires)


def apply_optional_noise(
    noisy: bool,
    depolarizing_prob: float,
    amplitude_damping_prob: float,
    num_wires: int,
):
    """
    Apply optional noise channels to each qubit after the ansatz.

    This should be called **inside** a QNode, *after* the ansatz circuit.
    """
    if not noisy:
        return

    for w in range(num_wires):
        if depolarizing_prob > 0:
            qml.DepolarizingChannel(depolarizing_prob, wires=w)
        if amplitude_damping_prob > 0:
            qml.AmplitudeDamping(amplitude_damping_prob, wires=w)


# ================================================================
# ANSATZ CONSTRUCTION
# ================================================================
def _call_ansatz(ansatz_fn, params, wires, symbols=None, coordinates=None, basis: str | None = None):
    """
    Call an ansatz function, automatically forwarding only the arguments it supports.

    This ensures compatibility between different ansatz definitions
    (e.g., UCCSD which expects molecular arguments, vs hardware-efficient layers).
    """
    sig = inspect.signature(ansatz_fn).parameters
    kwargs = {}
    if "symbols" in sig:
        kwargs["symbols"] = symbols
    if "coordinates" in sig:
        kwargs["coordinates"] = coordinates
    if "basis" in sig and basis is not None:
        kwargs["basis"] = basis
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
    Return a tuple `(ansatz_fn, init_params)` for the specified ansatz.

    Args:
        ansatz_name: Name of the ansatz (see vqe.ansatz.ANSATZES).
        num_wires: Number of qubits/wires.
        seed: Random seed for deterministic initialization.
        symbols, coordinates, basis: Optional molecule info.
        requires_grad: Whether parameters require gradients.
        scale: Initialization scale (for random ansatz types).
    """
    ansatz_fn = get_ansatz(ansatz_name)
    params = init_params(
        ansatz_name,
        num_wires,
        scale=scale,
        requires_grad=requires_grad,
        symbols=symbols,
        coordinates=coordinates,
        basis=basis,
        seed=seed,
    )
    return ansatz_fn, params


# ================================================================
# OPTIMIZER BUILDER
# ================================================================
def build_optimizer(optimizer_name: str, stepsize: float):
    """Return a PennyLane optimizer instance by name with the given stepsize."""
    return get_optimizer(optimizer_name, stepsize=stepsize)


# ================================================================
# QNODE CONSTRUCTION
# ================================================================
def make_energy_qnode(
    H,
    dev,
    ansatz_fn,
    num_wires,
    *,
    noisy=False,
    depolarizing_prob=0.0,
    amplitude_damping_prob=0.0,
    symbols=None,
    coordinates=None,
    basis="sto-3g",
    diff_method: str | None = None,
):
    """
    Build a QNode that computes the expectation value <H> for given parameters.

    Args:
        H: Hamiltonian (qml.Hamiltonian)
        dev: PennyLane device
        ansatz_fn: Callable ansatz circuit
        num_wires: Number of qubits
        noisy: Whether to include noise
        depolarizing_prob, amplitude_damping_prob: Noise probabilities
        symbols, coordinates, basis: Molecular info (for UCC ansatz)
        diff_method: Differentiation method override
    """
    if diff_method is None:
        diff_method = "finite-diff" if noisy else "parameter-shift"

    @qml.qnode(dev, diff_method=diff_method)
    def energy(params):
        _call_ansatz(ansatz_fn, params, range(num_wires), symbols, coordinates, basis)
        apply_optional_noise(noisy, depolarizing_prob, amplitude_damping_prob, num_wires)
        return qml.expval(H)

    return energy


def make_overlap00_fn(
    dev,
    ansatz_fn,
    num_wires,
    *,
    noisy=False,
    depolarizing_prob=0.0,
    amplitude_damping_prob=0.0,
    symbols=None,
    coordinates=None,
    basis="sto-3g",
    diff_method: str | None = None,
):
    """
    Construct a function overlap00(p_i, p_j) ≈ |⟨ψ_i|ψ_j⟩|².

    Uses the "adjoint trick":
        1. Prepare |ψ_i⟩
        2. Apply adjoint(ansatz)(p_j)
        3. Measure probability of |00...0⟩

    Returns:
        Callable overlap00(p_i, p_j) → float
    """
    if diff_method is None:
        diff_method = "finite-diff" if noisy else "parameter-shift"

    def _apply(params):
        _call_ansatz(ansatz_fn, params, range(num_wires), symbols, coordinates, basis)
        apply_optional_noise(noisy, depolarizing_prob, amplitude_damping_prob, num_wires)

    @qml.qnode(dev, diff_method=diff_method)
    def _overlap(p_i, p_j):
        _apply(p_i)
        qml.adjoint(_apply)(p_j)
        return qml.probs(wires=range(num_wires))

    def overlap00(p_i, p_j):
        return _overlap(p_i, p_j)[0]

    return overlap00
