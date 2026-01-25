# tests/test_run_signature_deterministic.py

from __future__ import annotations

import copy

import numpy as np


def test_vqe_run_signature_stable_across_semantic_equivalents() -> None:
    from vqe.io_utils import run_signature as vqe_run_signature

    cfg = {
        "molecule": "H2",
        "symbols": ["H", "H"],
        "geometry": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
        "basis": "sto-3g",
        "ansatz": "UCCSD",
        "optimizer": {"name": "Adam", "stepsize": 0.2, "iterations_planned": 50},
        "optimizer_name": "Adam",
        "seed": 0,
        "noisy": False,
        "depolarizing_prob": 0.0,
        "amplitude_damping_prob": 0.0,
        "mapping": "jordan_wigner",
        "meta": {
            "float_py": 0.30000000000000004,  # common FP artifact
            "float_np": np.float64(0.3),
            "int_np": np.int64(7),
        },
    }

    sig1 = vqe_run_signature(cfg)

    cfg2 = copy.deepcopy(cfg)
    cfg2["geometry"] = cfg["geometry"].tolist()  # list-of-lists vs ndarray
    cfg2["meta"]["float_py"] = 0.3
    cfg2["meta"]["float_np"] = 0.30000000000000004
    cfg2["meta"]["int_np"] = int(cfg2["meta"]["int_np"])

    sig2 = vqe_run_signature(cfg2)

    assert sig1 == sig2


def test_qite_run_signature_stable_across_semantic_equivalents() -> None:
    from qite.io_utils import run_signature as qite_run_signature

    cfg = {
        "molecule": "H2",
        "symbols": ["H", "H"],
        "coordinates": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
        "basis": "sto-3g",
        "seed": 0,
        "mapping": "jordan_wigner",
        "noisy": False,
        "depolarizing_prob": 0.0,
        "amplitude_damping_prob": 0.0,
        "noise_model_name": None,
        "dtau": 0.2,
        "steps": 50,
        "ansatz": "UCCSD",
        "varqite": {
            "fd_eps": np.float64(1e-3),
            "reg": 1e-6,
            "solver": "solve",
            "pinv_rcond": 1e-10,
        },
        "meta": {
            "float_py": 0.30000000000000004,
            "float_np": np.float64(0.3),
            "int_np": np.int64(7),
        },
    }

    sig1 = qite_run_signature(cfg)

    cfg2 = copy.deepcopy(cfg)
    cfg2["coordinates"] = cfg["coordinates"].tolist()
    cfg2["varqite"]["fd_eps"] = float(cfg2["varqite"]["fd_eps"])
    cfg2["meta"]["float_py"] = 0.3
    cfg2["meta"]["float_np"] = 0.30000000000000004
    cfg2["meta"]["int_np"] = int(cfg2["meta"]["int_np"])

    sig2 = qite_run_signature(cfg2)

    assert sig1 == sig2
