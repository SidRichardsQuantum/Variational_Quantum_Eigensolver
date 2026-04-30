"""
common.benchmarks
=================

Reusable helpers for benchmark-style notebook analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pennylane as qml

from common.environment import environment_metadata
from common.hamiltonian import build_hamiltonian
from common.persist import atomic_write_json

HARTREE_TO_EV = 27.211386245988

BENCHMARK_ROW_COLUMNS = [
    "benchmark_id",
    "question",
    "system",
    "system_type",
    "method",
    "mapping",
    "basis",
    "charge",
    "multiplicity",
    "active_electrons",
    "active_orbitals",
    "num_qubits",
    "hamiltonian_terms",
    "ansatz",
    "optimizer",
    "steps",
    "stepsize",
    "seed",
    "shots",
    "noise_model",
    "noise_level",
    "energy",
    "exact_energy",
    "abs_error",
    "runtime_s",
    "compute_runtime_s",
    "cache_hit",
    "status",
    "failure_reason",
]


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


def _row_from_result(
    *,
    benchmark_id: str,
    question: str,
    system: str,
    system_type: str,
    method: str,
    result: dict[str, Any],
    exact_energy: float,
    hamiltonian_terms: int,
    ansatz: str | None = None,
    optimizer: str | None = None,
    steps: int | None = None,
    stepsize: float | None = None,
    seed: int | None = None,
    shots: int | None = None,
    mapping: str | None = None,
    basis: str | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    active_electrons: int | None = None,
    active_orbitals: int | None = None,
) -> dict[str, Any]:
    energy = float(result["energy"])
    return {
        "benchmark_id": benchmark_id,
        "question": question,
        "system": system,
        "system_type": system_type,
        "method": method,
        "mapping": "" if mapping is None else str(mapping),
        "basis": "" if basis is None else str(basis),
        "charge": "" if charge is None else int(charge),
        "multiplicity": "" if multiplicity is None else int(multiplicity),
        "active_electrons": ("" if active_electrons is None else int(active_electrons)),
        "active_orbitals": "" if active_orbitals is None else int(active_orbitals),
        "num_qubits": int(result["num_qubits"]),
        "hamiltonian_terms": int(hamiltonian_terms),
        "ansatz": "" if ansatz is None else str(ansatz),
        "optimizer": "" if optimizer is None else str(optimizer),
        "steps": "" if steps is None else int(steps),
        "stepsize": "" if stepsize is None else float(stepsize),
        "seed": "" if seed is None else int(seed),
        "shots": "analytic" if shots is None else int(shots),
        "noise_model": "",
        "noise_level": "",
        "energy": energy,
        "exact_energy": float(exact_energy),
        "abs_error": abs(energy - float(exact_energy)),
        "runtime_s": float(result.get("runtime_s", 0.0)),
        "compute_runtime_s": float(result.get("compute_runtime_s", 0.0)),
        "cache_hit": bool(result.get("cache_hit", False)),
        "status": "ok",
        "failure_reason": "",
    }


def _exact_ground_and_term_count(
    hamiltonian: Any, num_qubits: int
) -> tuple[float, int]:
    matrix = np.asarray(
        qml.matrix(hamiltonian, wire_order=list(range(int(num_qubits)))),
        dtype=complex,
    )
    exact = float(np.min(np.linalg.eigvalsh(matrix)).real)
    return exact, int(len(hamiltonian))


def _expert_z_cross_method_suite(
    *,
    force: bool,
    suppress_stdout: bool,
) -> dict[str, Any]:
    benchmark_id = "expert-z-cross-method"
    question = (
        "Do VQE, VarQITE, and QPE run through one expert-mode benchmark contract?"
    )
    system = "single_qubit_pauli_z"
    exact_energy = -1.0
    hamiltonian = qml.Hamiltonian([1.0], [qml.PauliZ(0)])
    reference_state = [1]
    rows: list[dict[str, Any]] = []

    from qite import run_qite
    from qpe import run_qpe
    from vqe import run_vqe

    vqe_steps = 2
    vqe_stepsize = 0.1
    seed = 0
    vqe_result, _ = timed_call(
        run_vqe,
        suppress_stdout=suppress_stdout,
        molecule=benchmark_id,
        hamiltonian=hamiltonian,
        num_qubits=1,
        reference_state=reference_state,
        ansatz_name="RY-CZ",
        optimizer_name="Adam",
        steps=vqe_steps,
        stepsize=vqe_stepsize,
        seed=seed,
        plot=False,
        force=force,
    )
    rows.append(
        _row_from_result(
            benchmark_id=benchmark_id,
            question=question,
            system=system,
            system_type="model",
            method="VQE",
            result=vqe_result,
            exact_energy=exact_energy,
            hamiltonian_terms=1,
            ansatz="RY-CZ",
            optimizer="Adam",
            steps=vqe_steps,
            stepsize=vqe_stepsize,
            seed=seed,
        )
    )

    qite_steps = 2
    qite_result, _ = timed_call(
        run_qite,
        suppress_stdout=suppress_stdout,
        molecule=benchmark_id,
        hamiltonian=hamiltonian,
        num_qubits=1,
        reference_state=reference_state,
        ansatz_name="RY-CZ",
        steps=qite_steps,
        dtau=0.05,
        seed=seed,
        plot=False,
        show=False,
        force=force,
    )
    rows.append(
        _row_from_result(
            benchmark_id=benchmark_id,
            question=question,
            system=system,
            system_type="model",
            method="VarQITE",
            result=qite_result,
            exact_energy=exact_energy,
            hamiltonian_terms=1,
            ansatz="RY-CZ",
            steps=qite_steps,
            stepsize=0.05,
            seed=seed,
        )
    )

    qpe_result, _ = timed_call(
        run_qpe,
        suppress_stdout=suppress_stdout,
        molecule=benchmark_id,
        hamiltonian=hamiltonian,
        hf_state=reference_state,
        system_qubits=1,
        n_ancilla=2,
        t=1.0,
        trotter_steps=1,
        shots=None,
        seed=seed,
        plot=False,
        force=force,
    )
    rows.append(
        _row_from_result(
            benchmark_id=benchmark_id,
            question=question,
            system=system,
            system_type="model",
            method="QPE",
            result=qpe_result,
            exact_energy=exact_energy,
            hamiltonian_terms=1,
            seed=seed,
            shots=None,
        )
    )

    return {
        "suite": {
            "id": benchmark_id,
            "title": "Expert-Mode Single-Qubit Cross-Method Smoke Benchmark",
            "question": question,
            "scope": "Fast deterministic model-Hamiltonian benchmark for artifact plumbing.",
        },
        "rows": rows,
    }


def _h2_cross_method_suite(
    *,
    force: bool,
    suppress_stdout: bool,
) -> dict[str, Any]:
    benchmark_id = "h2-cross-method"
    question = (
        "How do VQE, VarQITE, and QPE compare on the canonical H2 STO-3G problem?"
    )
    system = "H2"
    mapping = "jordan_wigner"
    basis = "sto-3g"
    charge = 0
    multiplicity = 1
    seed = 0

    H, num_qubits, _hf_state = build_hamiltonian(
        molecule=system,
        mapping=mapping,
    )
    exact_energy, hamiltonian_terms = _exact_ground_and_term_count(H, int(num_qubits))
    rows: list[dict[str, Any]] = []

    from qite import run_qite
    from qpe import run_qpe
    from vqe import run_vqe

    vqe_steps = 25
    vqe_result, _ = timed_call(
        run_vqe,
        suppress_stdout=suppress_stdout,
        molecule=system,
        mapping=mapping,
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        steps=vqe_steps,
        seed=seed,
        plot=False,
        force=force,
    )
    rows.append(
        _row_from_result(
            benchmark_id=benchmark_id,
            question=question,
            system=system,
            system_type="molecule",
            method="VQE",
            result=vqe_result,
            exact_energy=exact_energy,
            hamiltonian_terms=hamiltonian_terms,
            ansatz="UCCSD",
            optimizer="Adam",
            steps=vqe_steps,
            seed=seed,
            mapping=mapping,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
        )
    )

    qite_steps = 25
    qite_result, _ = timed_call(
        run_qite,
        suppress_stdout=suppress_stdout,
        molecule=system,
        mapping=mapping,
        ansatz_name="UCCSD",
        steps=qite_steps,
        dtau=0.2,
        seed=seed,
        plot=False,
        show=False,
        force=force,
    )
    rows.append(
        _row_from_result(
            benchmark_id=benchmark_id,
            question=question,
            system=system,
            system_type="molecule",
            method="VarQITE",
            result=qite_result,
            exact_energy=exact_energy,
            hamiltonian_terms=hamiltonian_terms,
            ansatz="UCCSD",
            steps=qite_steps,
            stepsize=0.2,
            seed=seed,
            mapping=mapping,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
        )
    )

    qpe_result, _ = timed_call(
        run_qpe,
        suppress_stdout=suppress_stdout,
        molecule=system,
        mapping=mapping,
        n_ancilla=2,
        t=4.0,
        trotter_steps=1,
        shots=None,
        seed=seed,
        plot=False,
        force=force,
    )
    rows.append(
        _row_from_result(
            benchmark_id=benchmark_id,
            question=question,
            system=system,
            system_type="molecule",
            method="QPE",
            result=qpe_result,
            exact_energy=exact_energy,
            hamiltonian_terms=hamiltonian_terms,
            seed=seed,
            shots=None,
            mapping=mapping,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
        )
    )

    return {
        "suite": {
            "id": benchmark_id,
            "title": "H2 Cross-Method Benchmark",
            "question": question,
            "scope": (
                "Small chemistry benchmark comparing VQE, VarQITE, and QPE "
                "against exact diagonalization for H2/STO-3G."
            ),
        },
        "rows": rows,
    }


_BENCHMARK_SUITE_METADATA: dict[str, dict[str, str]] = {
    "expert-z-cross-method": {
        "id": "expert-z-cross-method",
        "title": "Expert-Mode Single-Qubit Cross-Method Smoke Benchmark",
        "question": (
            "Do VQE, VarQITE, and QPE run through one expert-mode "
            "benchmark contract?"
        ),
    },
    "h2-cross-method": {
        "id": "h2-cross-method",
        "title": "H2 Cross-Method Benchmark",
        "question": (
            "How do VQE, VarQITE, and QPE compare on the canonical H2 "
            "STO-3G problem?"
        ),
    },
}

_BENCHMARK_SUITE_RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "expert-z-cross-method": _expert_z_cross_method_suite,
    "h2-cross-method": _h2_cross_method_suite,
}


def list_benchmark_suites() -> list[dict[str, str]]:
    """
    Return the registered one-command benchmark suites.
    """
    return [
        dict(_BENCHMARK_SUITE_METADATA[suite_id])
        for suite_id in sorted(_BENCHMARK_SUITE_METADATA)
    ]


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return value
    return json.dumps(value, sort_keys=True)


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted(
        {key for row in rows for key in row if key not in BENCHMARK_ROW_COLUMNS}
    )
    columns = [*BENCHMARK_ROW_COLUMNS, *extra_columns]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _csv_value(row.get(col, "")) for col in columns})


def _markdown_rows_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "method",
        "system",
        "energy",
        "exact_energy",
        "abs_error",
        "runtime_s",
        "compute_runtime_s",
        "cache_hit",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    path: Path, payload: dict[str, Any], manifest: dict[str, Any]
) -> None:
    suite = payload["suite"]
    rows = payload["rows"]
    lines = [
        f"# {suite['title']}",
        "",
        f"Suite ID: `{suite['id']}`",
        "",
        f"Question: {suite['question']}",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Results",
        "",
        _markdown_rows_table(rows),
        "",
        "## Artifacts",
        "",
        f"- JSON: `{manifest['artifacts']['json']}`",
        f"- CSV: `{manifest['artifacts']['csv']}`",
        f"- Markdown report: `{manifest['artifacts']['markdown']}`",
        f"- Manifest: `{manifest['artifacts']['manifest']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_benchmark_suite(
    suite_id: str,
    *,
    out_dir: str | Path = "benchmark_runs",
    force: bool = True,
    suppress_stdout: bool = True,
) -> dict[str, Any]:
    """
    Run a registered benchmark suite and write JSON, CSV, Markdown, and manifest files.
    """
    suite_key = str(suite_id).strip()
    if suite_key not in _BENCHMARK_SUITE_RUNNERS:
        available = ", ".join(sorted(_BENCHMARK_SUITE_RUNNERS))
        raise ValueError(
            f"Unknown benchmark suite {suite_id!r}. Available: {available}"
        )

    suite_out = Path(out_dir) / suite_key
    suite_out.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    payload = _BENCHMARK_SUITE_RUNNERS[suite_key](
        force=bool(force),
        suppress_stdout=bool(suppress_stdout),
    )
    payload["schema_version"] = 1
    payload["generated_at"] = generated_at
    payload["environment"] = environment_metadata()

    json_path = suite_out / "results.json"
    csv_path = suite_out / "results.csv"
    md_path = suite_out / "report.md"
    manifest_path = suite_out / "manifest.json"

    atomic_write_json(json_path, payload)
    _write_rows_csv(csv_path, payload["rows"])

    manifest = {
        "schema_version": 1,
        "suite": payload["suite"],
        "generated_at": generated_at,
        "environment": payload["environment"],
        "artifacts": {
            "json": json_path.name,
            "csv": csv_path.name,
            "markdown": md_path.name,
            "manifest": manifest_path.name,
        },
        "row_count": len(payload["rows"]),
    }
    _write_report(md_path, payload, manifest)
    atomic_write_json(manifest_path, manifest)

    return {
        "suite": payload["suite"],
        "rows": payload["rows"],
        "artifacts": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "manifest": str(manifest_path),
        },
    }


def _resolve_results_json(path: str | Path) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"Benchmark results file not found: {p}")
    return p


def _comparison_row_key(row: dict[str, Any]) -> str:
    parts = [
        row.get("benchmark_id", ""),
        row.get("system", ""),
        row.get("method", ""),
        row.get("seed", ""),
        row.get("shots", ""),
        row.get("noise_model", ""),
        row.get("noise_level", ""),
    ]
    return "|".join(str(part) for part in parts)


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_benchmark_runs(
    base: str | Path,
    head: str | Path,
    *,
    energy_tol: float = 1e-8,
    abs_error_tol: float = 1e-8,
    runtime_ratio: float = 2.0,
) -> dict[str, Any]:
    """
    Compare two benchmark suite outputs and flag numerical/runtime drift.

    ``base`` and ``head`` may be either suite directories or direct paths to
    ``results.json`` files. Runtime checks use ``compute_runtime_s`` so cache
    hits do not look artificially faster than fresh runs.
    """
    base_path = _resolve_results_json(base)
    head_path = _resolve_results_json(head)
    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    head_payload = json.loads(head_path.read_text(encoding="utf-8"))

    base_rows = {_comparison_row_key(row): row for row in base_payload.get("rows", [])}
    head_rows = {_comparison_row_key(row): row for row in head_payload.get("rows", [])}

    base_keys = set(base_rows)
    head_keys = set(head_rows)
    missing = sorted(base_keys - head_keys)
    added = sorted(head_keys - base_keys)
    changes: list[dict[str, Any]] = []

    for key in sorted(base_keys & head_keys):
        base_row = base_rows[key]
        head_row = head_rows[key]

        for field, tolerance in (
            ("energy", float(energy_tol)),
            ("abs_error", float(abs_error_tol)),
        ):
            base_value = _float_or_none(base_row.get(field))
            head_value = _float_or_none(head_row.get(field))
            if base_value is None or head_value is None:
                continue
            delta = head_value - base_value
            if abs(delta) > tolerance:
                changes.append(
                    {
                        "key": key,
                        "field": field,
                        "base": base_value,
                        "head": head_value,
                        "delta": delta,
                        "tolerance": tolerance,
                        "status": "failed",
                    }
                )

        base_runtime = _float_or_none(base_row.get("compute_runtime_s"))
        head_runtime = _float_or_none(head_row.get("compute_runtime_s"))
        if (
            base_runtime is not None
            and head_runtime is not None
            and base_runtime > 0.0
            and head_runtime > base_runtime * float(runtime_ratio)
        ):
            changes.append(
                {
                    "key": key,
                    "field": "compute_runtime_s",
                    "base": base_runtime,
                    "head": head_runtime,
                    "ratio": head_runtime / base_runtime,
                    "tolerance": float(runtime_ratio),
                    "status": "failed",
                }
            )

    passed = not missing and not added and not changes
    return {
        "schema_version": 1,
        "base": str(base_path),
        "head": str(head_path),
        "base_suite": base_payload.get("suite", {}),
        "head_suite": head_payload.get("suite", {}),
        "thresholds": {
            "energy_tol": float(energy_tol),
            "abs_error_tol": float(abs_error_tol),
            "runtime_ratio": float(runtime_ratio),
        },
        "row_count_base": len(base_rows),
        "row_count_head": len(head_rows),
        "missing_rows": missing,
        "added_rows": added,
        "changes": changes,
        "passed": bool(passed),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m common.benchmarks",
        description="Run registered reproducible benchmark suites.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List available benchmark suites.")

    run_parser = sub.add_parser("run", help="Run one benchmark suite.")
    run_parser.add_argument("--suite", required=True, help="Benchmark suite id.")
    run_parser.add_argument(
        "--out",
        default="benchmark_runs",
        help="Output directory for benchmark artifacts.",
    )
    run_parser.add_argument(
        "--no-force",
        action="store_true",
        help="Allow cache reuse instead of forcing fresh solver runs.",
    )
    run_parser.add_argument(
        "--show-stdout",
        action="store_true",
        help="Show solver stdout while running the suite.",
    )

    compare_parser = sub.add_parser(
        "compare", help="Compare two benchmark suite outputs."
    )
    compare_parser.add_argument(
        "--base",
        required=True,
        help="Baseline suite directory or results.json path.",
    )
    compare_parser.add_argument(
        "--head",
        required=True,
        help="Candidate suite directory or results.json path.",
    )
    compare_parser.add_argument(
        "--energy-tol",
        type=float,
        default=1e-8,
        help="Allowed absolute energy delta.",
    )
    compare_parser.add_argument(
        "--abs-error-tol",
        type=float,
        default=1e-8,
        help="Allowed absolute-error delta.",
    )
    compare_parser.add_argument(
        "--runtime-ratio",
        type=float,
        default=2.0,
        help="Allowed head/base compute runtime ratio.",
    )
    compare_parser.add_argument(
        "--out",
        default=None,
        help="Optional path for the JSON comparison report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for suite in list_benchmark_suites():
            print(f"{suite['id']}\t{suite['title']}")
        return 0

    if args.command == "run":
        result = run_benchmark_suite(
            args.suite,
            out_dir=args.out,
            force=not bool(args.no_force),
            suppress_stdout=not bool(args.show_stdout),
        )
        print(f"Ran suite: {result['suite']['id']}")
        for kind, path in result["artifacts"].items():
            print(f"{kind}: {path}")
        return 0

    if args.command == "compare":
        result = compare_benchmark_runs(
            args.base,
            args.head,
            energy_tol=float(args.energy_tol),
            abs_error_tol=float(args.abs_error_tol),
            runtime_ratio=float(args.runtime_ratio),
        )
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.out is not None:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if bool(result["passed"]) else 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
