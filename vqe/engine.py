# vqe/engine.py
from __future__ import annotations
import inspect
import pennylane as qml
from pennylane import numpy as np

from .ansatz import get_ansatz, init_params
from .optimizer import get_optimizer


# ---------- Device & noise ----------

def make_device(num_wires: int, noisy: bool = False):
    """Return a PennyLane device with/without noise support."""
    dev_name = "default.mixed" if noisy else "default.qubit"
    return qml.device(dev_name, wires=num_wires)

def apply_optional_noise(noisy: bool, depolarizing_prob: float, amplitude_damping_prob: float, num_wires: int):
    """Insert noise channels if requested. Call this inside the qnode, after the ansatz."""
    if not noisy:
        return
    for w in range(num_wires):
        if depolarizing_prob > 0:
            qml.DepolarizingChannel(depolarizing_prob, wires=w)
        if amplitude_damping_prob > 0:
            qml.AmplitudeDamping(amplitude_damping_prob, wires=w)


# ---------- Ansatz plumbing ----------

def _call_ansatz(ansatz_fn, params, wires, symbols=None, coordinates=None, basis: str | None = None):
    """Call an ansatz function, passing only the kwargs it actually accepts."""
    sig = inspect.signature(ansatz_fn).parameters
    kwargs = {}
    if "symbols" in sig: kwargs["symbols"] = symbols
    if "coordinates" in sig: kwargs["coordinates"] = coordinates
    if "basis" in sig and basis is not None: kwargs["basis"] = basis
    return ansatz_fn(params, wires=wires, **kwargs)

def build_ansatz(ansatz_name: str, num_wires: int, *, seed: int = 0,
                 symbols=None, coordinates=None, basis: str = "sto-3g",
                 requires_grad: bool = True, scale: float = 0.01):
    """
    Return (ansatz_fn, init_params) for the named ansatz, with molecule-aware
    initialization when needed (e.g., UCCSD).
    """
    ansatz_fn = get_ansatz(ansatz_name)
    params = init_params(ansatz_name, num_wires,
                         scale=scale, requires_grad=requires_grad,
                         symbols=symbols, coordinates=coordinates, basis=basis, seed=seed)
    return ansatz_fn, params


# ---------- Optimizer ----------

def build_optimizer(optimizer_name: str, stepsize: float):
    """Return a PennyLane optimizer instance by name with the given stepsize."""
    return get_optimizer(optimizer_name, stepsize=stepsize)


# ---------- QNodes ----------

def make_energy_qnode(H, dev, ansatz_fn, num_wires, *, noisy=False,
                      depolarizing_prob=0.0, amplitude_damping_prob=0.0,
                      symbols=None, coordinates=None, basis="sto-3g",
                      diff_method: str | None = None):
    """
    Build a QNode that returns <H> for given params under the chosen ansatz, with optional noise.
    """
    if diff_method is None:
        diff_method = "finite-diff" if noisy else "parameter-shift"

    @qml.qnode(dev, diff_method=diff_method)
    def energy(params):
        _call_ansatz(ansatz_fn, params, range(num_wires), symbols, coordinates, basis)
        apply_optional_noise(noisy, depolarizing_prob, amplitude_damping_prob, num_wires)
        return qml.expval(H)

    return energy


def make_overlap00_fn(dev, ansatz_fn, num_wires, *, noisy=False,
                      depolarizing_prob=0.0, amplitude_damping_prob=0.0,
                      symbols=None, coordinates=None, basis="sto-3g",
                      diff_method: str | None = None):
    """
    Return a function overlap00(p_i, p_j) ≈ |<ψ_i|ψ_j>|^2 using the adjoint trick and probs()[0].
    Works for *any* ansatz (including UCC* that prepare HF internally).
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
