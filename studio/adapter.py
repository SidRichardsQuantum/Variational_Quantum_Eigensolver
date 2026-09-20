"""Catalogue, strict experiment validation, and the scientific API boundary."""

import inspect
import json
import math

from common.molecules import MOLECULES
from common.persist import to_serializable
from qite.core import run_qite
from vqe.adapt import run_adapt_vqe
from vqe.ansatz import ANSATZES
from vqe.core import run_vqe
from vqe.optimizer import OPTIMIZERS

DEFAULTS = {k: p.default for k, p in inspect.signature(run_vqe).parameters.items()}
ADAPT_DEFAULTS = {
    k: p.default for k, p in inspect.signature(run_adapt_vqe).parameters.items()
}
QITE_DEFAULTS = {
    k: p.default for k, p in inspect.signature(run_qite).parameters.items()
}
METHODS = {"vqe": "Ground-state VQE", "adapt_vqe": "ADAPT-VQE", "varqite": "VarQITE"}
API_NAMES = {"ansatz": "ansatz_name", "optimizer": "optimizer_name"}


def catalogue():
    def method_fields(method):
        defaults = {
            "vqe": DEFAULTS,
            "adapt_vqe": ADAPT_DEFAULTS,
            "varqite": QITE_DEFAULTS,
        }[method]

        def field(group, name, label, kind, help_text, **extra):
            default = defaults.get(API_NAMES.get(name, name))
            if name == "basis":
                default = MOLECULES[defaults["molecule"]]["basis"].lower()
            return dict(
                group=group,
                name=name,
                label=label,
                type=kind,
                default=default,
                help=help_text,
                **extra,
            )

        fields = [
            field(
                "problem",
                "molecule",
                "Molecule",
                "enum",
                "Geometry and chemistry come from the package registry.",
                values=list(MOLECULES),
            ),
            field(
                "problem",
                "basis",
                "Basis",
                "registry",
                "Registry-controlled molecular basis.",
                values_by_molecule={
                    k: v["basis"].lower() for k, v in MOLECULES.items()
                },
            ),
            field(
                "problem",
                "mapping",
                "Qubit mapping",
                "enum",
                "Fermion-to-qubit encoding.",
                values=["jordan_wigner", "parity", "bravyi_kitaev"],
            ),
        ]
        if method in {"vqe", "varqite"}:
            fields += [
                field(
                    "settings",
                    "ansatz",
                    "Ansatz",
                    "enum",
                    "Package circuit registry; compatibility is checked by the solver.",
                    values=list(ANSATZES),
                )
            ]
        else:
            fields += [
                field(
                    "settings",
                    "pool",
                    "Operator pool",
                    "enum",
                    "UCC excitation pool used to grow the ansatz.",
                    values=["uccsd", "uccs", "uccd"],
                ),
                field(
                    "settings",
                    "max_ops",
                    "Maximum operators",
                    "integer",
                    "Operator budget; zero evaluates the Hartree–Fock reference.",
                    minimum=0,
                ),
                field(
                    "settings",
                    "grad_tol",
                    "Gradient tolerance",
                    "number",
                    "Stop growing when the largest remaining pool gradient falls below this value.",
                    minimum=0,
                ),
            ]
        fields += [
            field(
                "settings",
                "optimizer",
                "Optimizer",
                "enum",
                "Classical energy minimizer.",
                values=list(OPTIMIZERS),
            )
        ]
        if method in {"vqe", "varqite"}:
            fields += [
                field(
                    "settings",
                    "steps",
                    "Optimizer steps",
                    "integer",
                    "Fixed iteration budget; completion does not certify convergence.",
                    minimum=0,
                ),
                field(
                    "settings",
                    "stepsize",
                    "Step size",
                    "number",
                    "Leave empty for the selected optimizer's calibrated default.",
                    nullable=True,
                    exclusive_minimum=0,
                ),
            ]
        else:
            fields += [
                field(
                    "settings",
                    "inner_steps",
                    "Inner optimizer steps",
                    "integer",
                    "Optimization budget at each outer iteration.",
                    minimum=0,
                ),
                field(
                    "settings",
                    "inner_stepsize",
                    "Inner step size",
                    "number",
                    "ADAPT's explicit package default; automatic step size is not supported by this API.",
                    exclusive_minimum=0,
                ),
            ]
        fields += [
            field(
                "settings",
                "seed",
                "Seed",
                "integer",
                "Random seed passed to the solver.",
                minimum=0,
                maximum=2**32 - 1,
            )
        ]
        if method in {"vqe", "varqite"}:
            fields += [
                field(
                    "settings",
                    "energy_tol",
                    "Energy-change tolerance (Ha)",
                    "number",
                    "Optional stopping criterion; not an accuracy certificate. Leave empty for a fixed budget.",
                    nullable=True,
                    minimum=0,
                ),
                field(
                    "settings",
                    "patience",
                    "Consecutive small changes",
                    "integer",
                    "Number of consecutive updates satisfying the energy-change tolerance.",
                    minimum=1,
                ),
            ]
        if method == "varqite":
            fields = [f for f in fields if f["name"] not in {"optimizer", "stepsize"}]
            next(f for f in fields if f["name"] == "steps")[
                "label"
            ] = "Imaginary-time steps"
            fields += [
                field(
                    "settings",
                    "dtau",
                    "Imaginary-time step",
                    "number",
                    "McLachlan update time step.",
                    exclusive_minimum=0,
                ),
                field(
                    "settings",
                    "solver",
                    "Linear solver",
                    "enum",
                    "Solver for the McLachlan system.",
                    values=["solve", "pinv", "lstsq"],
                ),
                field(
                    "settings",
                    "reg",
                    "Regularization",
                    "number",
                    "Diagonal regularization of the metric.",
                    minimum=0,
                ),
                field(
                    "settings",
                    "fd_eps",
                    "Finite-difference displacement",
                    "number",
                    "State derivative displacement.",
                    exclusive_minimum=0,
                ),
                field(
                    "settings",
                    "pinv_rcond",
                    "Pseudoinverse cutoff",
                    "number",
                    "Relative singular-value cutoff.",
                    exclusive_minimum=0,
                ),
            ]
        return fields

    return {
        "schema_version": 1,
        "methods": [
            {"id": key, "label": label, "fields": method_fields(key)}
            for key, label in METHODS.items()
        ],
        "optimizer_stepsizes": {k: v["stepsize"] for k, v in OPTIMIZERS.items()},
    }


def normalize(raw):
    if not isinstance(raw, dict) or set(raw) - {
        "method",
        "problem",
        "settings",
        "refinement",
    }:
        raise ValueError("Expected an experiment object with method, problem, settings")
    method = raw.get("method", "vqe")
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("Unsupported method; choose a catalogue entry")
    fields = next(m["fields"] for m in catalogue()["methods"] if m["id"] == method)
    out = {"method": method, "problem": {}, "settings": {}}
    for group in ("problem", "settings"):
        values = raw.get(group, {})
        allowed = {f["name"] for f in fields if f["group"] == group}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError(f"Unknown fields or invalid object in {group}")
    for f in fields:
        value = raw.get(f["group"], {}).get(f["name"], f["default"])
        kind = f["type"]
        if kind == "registry":
            expected = f["values_by_molecule"][out["problem"]["molecule"]]
            value = raw.get("problem", {}).get("basis", expected)
            if value != expected:
                raise ValueError(f"basis must be registry value {expected!r}")
        elif kind == "enum":
            if not isinstance(value, str) or value not in f["values"]:
                raise ValueError(f"Invalid {f['name']}: choose a catalogue value")
        elif value is None and f.get("nullable"):
            pass
        else:
            valid_type = (
                type(value) is int if kind == "integer" else type(value) in (int, float)
            )
            try:
                finite = valid_type and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"{f['name']} must be a finite {kind}")
            if (
                value < f.get("minimum", -math.inf)
                or value > f.get("maximum", math.inf)
                or value <= f.get("exclusive_minimum", -math.inf)
            ):
                raise ValueError(f"{f['name']} outside allowed range")
        out[f["group"]][f["name"]] = value
    if "refinement" in raw:
        ref = raw["refinement"]
        if (
            method != "varqite"
            or not isinstance(ref, dict)
            or set(ref) != {"artifact", "digest"}
            or not all(isinstance(v, str) and v for v in ref.values())
        ):
            raise ValueError(
                "refinement requires a VarQITE run and source artifact/digest"
            )
        out["refinement"] = dict(ref)
    return out


def to_kwargs(experiment):
    config = normalize(experiment)
    kwargs = {
        **config["problem"],
        **{API_NAMES.get(k, k): v for k, v in config["settings"].items()},
        "plot": False,
    }
    if "refinement" in config:
        kwargs.update(refinement_kwargs(config))
    return kwargs


def execute(experiment, *, progress_callback=None):
    config = normalize(experiment)
    runner = {"vqe": run_vqe, "adapt_vqe": run_adapt_vqe, "varqite": run_qite}[
        config["method"]
    ]
    kwargs = to_kwargs(config)
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    return runner(**kwargs)


def json_bytes(value):
    """Stable strict JSON preserving precision and rejecting non-finite values."""
    return json.dumps(
        to_serializable(value), sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def refinement_kwargs(config):
    """Verify an immutable VQE artifact before transferring its parameters."""
    import numpy as np

    from common.problem import problem_metadata, resolve_problem

    from .history import artifacts

    ref = config["refinement"]
    source = artifacts().get(ref["artifact"])
    if source is None or source["artifact_digest"] != ref["digest"]:
        raise ValueError("Refinement source is missing or changed")
    c, r = source["resolved_config"], source["result"]
    if source["method"] != "vqe" or source["status"] != "completed" or c.get("noise"):
        raise ValueError("Refinement requires a successful noiseless VQE artifact")
    if c.get("ansatz") != config["settings"]["ansatz"] or c.get("ansatz_kwargs"):
        raise ValueError(
            "Refinement ansatz and parameter ordering must match the source"
        )
    problem = resolve_problem(**config["problem"])
    target = problem_metadata(problem)
    if any(json_bytes(c.get(k)) != json_bytes(v) for k, v in target.items()):
        raise ValueError(
            "Refinement requires the same resolved geometry, active space, mapping and reference"
        )
    shape = r.get("final_params_shape")
    if not isinstance(shape, list):
        raise ValueError("Source does not record its parameter shape")
    params = np.asarray(r["final_params"], dtype=float).reshape(shape)
    if not np.all(np.isfinite(params)):
        raise ValueError("Source parameters are non-finite")
    from common.termination import supplied_parameters
    from qite.engine import build_ansatz, make_device, make_state_qnode

    ansatz, template = build_ansatz(
        config["settings"]["ansatz"],
        problem.num_qubits,
        symbols=problem.symbols,
        coordinates=problem.coordinates,
        basis=problem.basis,
        charge=problem.charge,
        multiplicity=problem.multiplicity,
        mapping=problem.mapping,
        active_electrons=problem.active_electrons,
        active_orbitals=problem.active_orbitals,
        hf_state=problem.reference_state,
    )
    supplied_parameters(params, template)
    prepared = np.asarray(
        make_state_qnode(make_device(problem.num_qubits), ansatz, problem.num_qubits)(
            params
        )
    )
    original = np.asarray(r["final_state_real"]) + 1j * np.asarray(
        r["final_state_imag"]
    )
    if original.shape != prepared.shape or not np.isclose(
        abs(np.vdot(original, prepared)), 1.0, atol=1e-8, rtol=0
    ):
        raise ValueError(
            "Refinement circuit preparation differs from the source VQE state"
        )
    provenance = {
        **ref,
        "energy": r["energy"],
        "compute_runtime_s": r.get("compute_runtime_s"),
    }
    return {"initial_params": params, "initialization_source": provenance}
