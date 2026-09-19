"""Catalogue, strict experiment validation, and the scientific API boundary."""

import inspect
import json
import math

from common.molecules import MOLECULES
from common.persist import to_serializable
from vqe.adapt import run_adapt_vqe
from vqe.ansatz import ANSATZES
from vqe.core import run_vqe
from vqe.optimizer import OPTIMIZERS

DEFAULTS = {k: p.default for k, p in inspect.signature(run_vqe).parameters.items()}
ADAPT_DEFAULTS = {
    k: p.default for k, p in inspect.signature(run_adapt_vqe).parameters.items()
}
METHODS = {"vqe": "Ground-state VQE", "adapt_vqe": "ADAPT-VQE"}
API_NAMES = {"ansatz": "ansatz_name", "optimizer": "optimizer_name"}


def catalogue():
    def method_fields(method):
        defaults = DEFAULTS if method == "vqe" else ADAPT_DEFAULTS

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
        if method == "vqe":
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
        if method == "vqe":
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
    if not isinstance(raw, dict) or set(raw) - {"method", "problem", "settings"}:
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
    return out


def to_kwargs(experiment):
    config = normalize(experiment)
    kwargs = {
        **config["problem"],
        **{API_NAMES.get(k, k): v for k, v in config["settings"].items()},
        "plot": False,
    }
    if config["method"] == "adapt_vqe":
        # Basis is descriptive registry metadata; ADAPT has no basis argument.
        kwargs.pop("basis")
    return kwargs


def execute(experiment, *, progress_callback=None):
    config = normalize(experiment)
    runner = run_vqe if config["method"] == "vqe" else run_adapt_vqe
    kwargs = to_kwargs(config)
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    return runner(**kwargs)


def json_bytes(value):
    """Stable strict JSON preserving precision and rejecting non-finite values."""
    return json.dumps(
        to_serializable(value), sort_keys=True, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
