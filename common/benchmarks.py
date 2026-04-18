"""
common.benchmarks
=================

Reusable helpers for benchmark-style notebook analysis.
"""

from __future__ import annotations

import time
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable

import numpy as np
import pennylane as qml

from common.hamiltonian import build_hamiltonian


def timed_call(
    fn: Callable[..., Any],
    /,
    *,
    suppress_stdout: bool = False,
    **kwargs,
) -> tuple[Any, float]:
    """
    Call a function and return (result, elapsed_seconds).

    This is intentionally lightweight so notebooks can measure API wall time
    without reimplementing timing boilerplate.
    """
    t0 = time.perf_counter()
    if suppress_stdout:
        sink = StringIO()
        with redirect_stdout(sink):
            result = fn(**kwargs)
    else:
        result = fn(**kwargs)
    return result, float(time.perf_counter() - t0)


def summary_stats(xs) -> dict[str, float]:
    """
    Mean/std/min/max summary for a numeric sequence.
    """
    arr = np.asarray(xs, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def exact_ground_energy_for_problem(**problem_spec) -> float:
    """
    Exact ground energy from the resolved qubit Hamiltonian for a problem spec.
    """
    H, n_qubits, _ = build_hamiltonian(**problem_spec)
    matrix = np.asarray(
        qml.matrix(H, wire_order=list(range(int(n_qubits)))),
        dtype=complex,
    )
    eigs = np.linalg.eigvalsh(matrix)
    return float(np.min(eigs).real)


def summarize_problem(**problem_spec) -> dict[str, object]:
    """
    Build one compact summary row for a resolved problem specification.
    """
    H, n_qubits, _ = build_hamiltonian(**problem_spec)
    matrix = np.asarray(
        qml.matrix(H, wire_order=list(range(int(n_qubits)))),
        dtype=complex,
    )
    eigs = np.linalg.eigvalsh(matrix)
    return {
        **problem_spec,
        "num_qubits": int(n_qubits),
        "hamiltonian_terms": int(len(H)),
        "exact_ground_energy": float(np.min(eigs).real),
    }
