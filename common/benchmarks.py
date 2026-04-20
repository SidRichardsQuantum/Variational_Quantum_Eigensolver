"""
common.benchmarks
=================

Reusable helpers for benchmark-style notebook analysis.
"""

from __future__ import annotations

import time
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from typing import Any, Callable

import numpy as np
import pennylane as qml

from common.hamiltonian import build_hamiltonian

HARTREE_TO_EV = 27.211386245988


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


def ionization_energy_panel(
    coverage_rows: list[dict[str, Any]],
    *,
    pairs: list[tuple[str, str]] | None = None,
) -> list[dict[str, object]]:
    """
    Build neutral/cation ionization-energy rows from registry coverage output.

    The input is intentionally the flat output from
    ``summarize_registry_coverage(...)`` so notebooks can build expensive
    Hamiltonians once and reuse the same metadata table for multiple views.
    """
    by_name = {str(row["molecule"]): row for row in coverage_rows}
    selected_pairs = (
        [
            (name, f"{name}+")
            for name, row in by_name.items()
            if int(row.get("charge", 0)) == 0 and f"{name}+" in by_name
        ]
        if pairs is None
        else [(str(neutral), str(cation)) for neutral, cation in pairs]
    )

    rows: list[dict[str, Any]] = []
    for neutral_name, cation_name in selected_pairs:
        if neutral_name not in by_name:
            raise ValueError(f"Missing neutral coverage row for {neutral_name!r}.")
        if cation_name not in by_name:
            raise ValueError(f"Missing cation coverage row for {cation_name!r}.")

        neutral = by_name[neutral_name]
        cation = by_name[cation_name]
        neutral_energy = float(neutral["exact_ground_energy"])
        cation_energy = float(cation["exact_ground_energy"])
        ionization_ha = cation_energy - neutral_energy

        rows.append(
            {
                "system": neutral_name,
                "neutral": neutral_name,
                "cation": cation_name,
                "neutral_exact_ground_energy": neutral_energy,
                "cation_exact_ground_energy": cation_energy,
                "ionization_energy_ha": float(ionization_ha),
                "ionization_energy_ev": float(ionization_ha * HARTREE_TO_EV),
                "neutral_num_electrons": int(neutral["num_electrons"]),
                "cation_num_electrons": int(cation["num_electrons"]),
                "neutral_num_qubits": int(neutral["num_qubits"]),
                "cation_num_qubits": int(cation["num_qubits"]),
                "neutral_hamiltonian_terms": int(neutral["hamiltonian_terms"]),
                "cation_hamiltonian_terms": int(cation["hamiltonian_terms"]),
            }
        )

    rows.sort(
        key=lambda row: (
            int(row["neutral_num_qubits"]),
            int(row["neutral_num_electrons"]),
            str(row["system"]),
        )
    )
    return rows


def qpe_branch_candidates(
    bitstring: str,
    t: float,
    ref_energy: float,
) -> dict[str, float]:
    """
    Return MSB-first and LSB-first phase/energy interpretations for a QPE bitstring.

    QPE notebooks often need to distinguish the displayed dominant bitstring from
    the phase-orientation branch used to convert that bitstring into an energy.
    Keeping this helper in ``common`` avoids duplicating that diagnostic logic.
    """
    from qpe.core import bitstring_to_phase, phase_to_energy_unwrapped

    phase_msb = bitstring_to_phase(str(bitstring), msb_first=True)
    phase_lsb = bitstring_to_phase(str(bitstring), msb_first=False)
    energy_msb = phase_to_energy_unwrapped(phase_msb, float(t), ref_energy=ref_energy)
    energy_lsb = phase_to_energy_unwrapped(phase_lsb, float(t), ref_energy=ref_energy)
    return {
        "phase_msb": float(phase_msb),
        "phase_lsb": float(phase_lsb),
        "energy_msb": float(energy_msb),
        "energy_lsb": float(energy_lsb),
    }


def analyze_qpe_result(
    result: dict[str, Any],
    exact_ground: float,
    *,
    good_enough_error_ha: float = 0.05,
    branch_tol_ha: float = 1e-9,
) -> dict[str, object]:
    """
    Summarize QPE result quality and identify branch/bin failure modes.

    The returned dictionary is intentionally flat so benchmark notebooks can add
    it directly to a pandas record row.
    """
    t = float(result["t"])
    hf_energy = float(result["hf_energy"])
    selected_energy = float(result["energy"])
    selected_bitstring = str(result["best_bitstring"])
    probs = {str(k): float(v) for k, v in result["probs"].items()}

    if selected_bitstring not in probs:
        raise ValueError("QPE result best_bitstring is missing from probs.")
    if not probs:
        raise ValueError("QPE result contains no probabilities.")

    exact_ground_f = float(exact_ground)
    selected = qpe_branch_candidates(selected_bitstring, t, hf_energy)
    selected_candidates = [selected["energy_msb"], selected["energy_lsb"]]
    oracle_selected_energy = min(
        selected_candidates,
        key=lambda energy: abs(float(energy) - exact_ground_f),
    )

    all_candidates = []
    for bitstring, probability in probs.items():
        candidates = qpe_branch_candidates(bitstring, t, hf_energy)
        for orientation in ("msb", "lsb"):
            energy = float(candidates[f"energy_{orientation}"])
            all_candidates.append(
                {
                    "bitstring": bitstring,
                    "orientation": orientation,
                    "probability": probability,
                    "phase": float(candidates[f"phase_{orientation}"]),
                    "energy": energy,
                    "abs_error": abs(energy - exact_ground_f),
                }
            )

    oracle_any = min(all_candidates, key=lambda row: row["abs_error"])
    selected_abs_error = abs(selected_energy - exact_ground_f)
    branch_selection_failure = (
        abs(float(oracle_selected_energy) - selected_energy) > float(branch_tol_ha)
        and abs(float(oracle_selected_energy) - exact_ground_f) + float(branch_tol_ha)
        < selected_abs_error
    )
    dominant_bin_failure = (
        float(oracle_any["abs_error"]) + float(branch_tol_ha) < selected_abs_error
    )
    resolution_or_alias_failure = float(oracle_any["abs_error"]) > float(
        good_enough_error_ha
    )

    period_ha = 2 * np.pi / t
    bin_width_phase = 1.0 / (2 ** int(result["n_ancilla"]))
    bin_width_energy_ha = period_ha * bin_width_phase

    return {
        "energy": selected_energy,
        "abs_error": selected_abs_error,
        "signed_error": selected_energy - exact_ground_f,
        "hf_energy": hf_energy,
        "best_bitstring": selected_bitstring,
        "best_probability": probs[selected_bitstring],
        "phase": float(result["phase"]),
        "period_ha": float(period_ha),
        "bin_width_phase": float(bin_width_phase),
        "bin_width_energy_ha": float(bin_width_energy_ha),
        "oracle_same_bitstring_energy": float(oracle_selected_energy),
        "oracle_same_bitstring_abs_error": abs(
            float(oracle_selected_energy) - exact_ground_f
        ),
        "oracle_any_bitstring": str(oracle_any["bitstring"]),
        "oracle_any_orientation": str(oracle_any["orientation"]),
        "oracle_any_energy": float(oracle_any["energy"]),
        "oracle_any_abs_error": float(oracle_any["abs_error"]),
        "branch_selection_failure": bool(branch_selection_failure),
        "dominant_bin_failure": bool(dominant_bin_failure),
        "resolution_or_alias_failure": bool(resolution_or_alias_failure),
    }


def qpe_calibration_plan(
    *,
    ancillas_grid,
    times_grid,
    trotter_grid,
    shots_grid,
    seeds,
) -> list[dict[str, object]]:
    """
    Build run rows for a QPE calibration grid.

    Analytic runs (``shots is None``) are deterministic, so they are scheduled
    once with seed 0. Finite-shot runs are repeated over the provided seeds.
    """
    rows: list[dict[str, object]] = []
    for ancillas, t, trotter_steps, shots in product(
        ancillas_grid,
        times_grid,
        trotter_grid,
        shots_grid,
    ):
        run_seeds = [0] if shots is None else list(seeds)
        for seed in run_seeds:
            rows.append(
                {
                    "ancillas": int(ancillas),
                    "t": float(t),
                    "trotter_steps": int(trotter_steps),
                    "shots": None if shots is None else int(shots),
                    "seed": int(seed),
                }
            )
    return rows


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
