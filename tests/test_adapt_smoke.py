# tests/test_adapt_smoke.py

from __future__ import annotations

import math
from typing import Any


from vqe.adapt import run_adapt_vqe


def _is_finite_number(x: Any) -> bool:
    try:
        return isinstance(x, (int, float)) and math.isfinite(float(x))
    except Exception:
        return False


def test_adapt_vqe_smoke_noiseless_h2_uccs(tmp_path, monkeypatch):
    # Isolate any filesystem I/O (results/, images/) to a temp directory.
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))

    import warnings

    warnings.filterwarnings(
        "ignore",
        message="Output seems independent of input.",
        category=UserWarning,
    )

    res = run_adapt_vqe(
        molecule="H2",
        pool="uccs",  # smallest chemistry pool; keeps this test cheap
        max_ops=4,
        grad_tol=1e-3,
        inner_steps=8,
        inner_stepsize=0.2,
        optimizer_name="Adam",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        noise_model=None,
        plot=False,  # tests should not pop GUI windows
        force=True,  # avoid any accidental reuse if env var is ignored
    )

    assert isinstance(res, dict)

    required_keys = {
        "energy",
        "energies",
        "inner_energies",
        "max_gradients",
        "selected_operators",
        "final_params",
        "num_qubits",
        "config",
    }
    missing = required_keys.difference(res.keys())
    assert not missing, f"Missing keys: {sorted(missing)}"

    assert isinstance(res["num_qubits"], int)
    assert res["num_qubits"] > 0

    assert _is_finite_number(res["energy"])

    energies = res["energies"]
    inner_energies = res["inner_energies"]
    max_grads = res["max_gradients"]
    selected_ops = res["selected_operators"]
    final_params = res["final_params"]
    cfg = res["config"]

    assert isinstance(energies, list) and len(energies) >= 1
    assert all(_is_finite_number(e) for e in energies)

    assert isinstance(inner_energies, list)
    assert len(inner_energies) == len(energies)
    assert all(isinstance(traj, list) and len(traj) >= 1 for traj in inner_energies)
    assert all(_is_finite_number(e) for traj in inner_energies for e in traj)

    assert isinstance(max_grads, list)
    # max_gradients are recorded for each scoring step; last outer iteration may stop without scoring
    assert len(max_grads) in {len(energies) - 1, len(energies)}
    assert all(_is_finite_number(g) for g in max_grads)

    assert isinstance(selected_ops, list)
    assert len(selected_ops) <= 4
    for op in selected_ops:
        assert isinstance(op, dict)
        assert op.get("kind") in {"single", "double"}
        wires = op.get("wires")
        assert isinstance(wires, list) and len(wires) in {2, 4}
        assert all(isinstance(w, int) for w in wires)

    assert isinstance(final_params, list)
    assert len(final_params) == len(selected_ops)
    assert all(_is_finite_number(t) for t in final_params)

    assert isinstance(cfg, dict)
    assert cfg.get("mapping") == "jordan_wigner"
    assert cfg.get("noisy") is False
    assert cfg.get("adapt_pool") == "uccs"
    assert cfg.get("adapt_max_ops") == 4
    assert abs(float(cfg.get("adapt_grad_tol")) - 1e-3) < 1e-15
    assert cfg.get("adapt_inner_steps") == 8
    assert abs(float(cfg.get("adapt_inner_stepsize")) - 0.2) < 1e-15
