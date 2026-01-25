# tests/test_cache_roundtrip.py

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict


def _call_flex(fn: Callable[..., Any], /, **kwargs: Any) -> Any:
    """
    Call a function with only the kwargs it actually accepts.
    This keeps the smoke test resilient to minor signature differences.
    """
    sig = inspect.signature(fn)
    accepted = {}
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())

    if has_var_kw:
        accepted = dict(kwargs)
    else:
        for k, v in kwargs.items():
            if k in sig.parameters:
                accepted[k] = v

    return fn(**accepted)


def test_vqe_cache_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))

    import vqe.io_utils as vqe_io

    cfg = vqe_io.make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        ansatz_desc="UCCSD",
        optimizer_name="Adam",
        stepsize=0.2,
        max_iterations=5,
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        molecule_label="H2",
    )

    h = vqe_io.run_signature(cfg)
    prefix = vqe_io.make_filename_prefix(
        cfg, noisy=False, seed=0, hash_str=h, algo="vqe"
    )

    record: Dict[str, Any] = {
        "config": cfg,
        "result": {
            "energy": -1.0,
            "energies": [-0.5, -0.8, -1.0],
            "steps": 2,
            "final_state_real": [1.0, 0.0],
            "final_state_imag": [0.0, 0.0],
            "num_qubits": 2,
        },
    }

    vqe_io.save_run_record(prefix, record)
    loaded = vqe_io.load_run_record(prefix)
    assert loaded is not None
    assert loaded == record


def test_qite_cache_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))

    import qite.io_utils as qite_io

    cfg = qite_io.make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        dtau=0.1,
        steps=10,
        molecule_label="H2",
        ansatz_name="UCCSD",
    )

    h = qite_io.run_signature(cfg)
    prefix = qite_io.make_filename_prefix(
        cfg, noisy=False, seed=0, hash_str=h, algo="qite"
    )

    record: Dict[str, Any] = {
        "config": cfg,
        "result": {
            "energy": -1.0,
            "energies": [-0.2, -0.7, -1.0],
            "steps": 2,
            "final_state_real": [1.0, 0.0],
            "final_state_imag": [0.0, 0.0],
            "num_qubits": 2,
        },
    }

    qite_io.save_run_record(prefix, record)
    loaded = qite_io.load_run_record(prefix)
    assert loaded is not None
    assert loaded == record


def test_qpe_cache_roundtrip(tmp_path, monkeypatch) -> None:
    """
    QPE caching is slightly different; this validates that (a) we can compute a cache key,
    (b) persist a JSON payload to the computed cache path, and (c) load it back.

    This is intentionally compute-free (no Hamiltonian/QNodes).
    """
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))

    import qpe.io_utils as qpe_io
    from common.persist import atomic_write_json, read_json

    # Minimal payload that resembles a QPE result artifact
    payload: Dict[str, Any] = {
        "molecule": "H2",
        "n_ancilla": 4,
        "t": 1.0,
        "seed": 0,
        "shots": 100,
        "trotter_steps": 1,
        "mapping": "jordan_wigner",
        "unit": "angstrom",
        "noise": {"p_dep": 0.0, "p_amp": 0.0},
        "probs": {"0000": 1.0},
    }

    key = qpe_io.signature_hash(
        molecule="H2",
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=100,
        noise={"p_dep": 0.0, "p_amp": 0.0},
        trotter_steps=1,
        mapping="jordan_wigner",
        unit="angstrom",
    )

    # Prefer the package's cache_path if available; otherwise write directly under results/qpe/.
    if hasattr(qpe_io, "cache_path") and callable(getattr(qpe_io, "cache_path")):
        path = qpe_io.cache_path(
            molecule="H2",
            n_ancilla=4,
            t=1.0,
            seed=0,
            noise={"p_dep": 0.0, "p_amp": 0.0},
            key=key,
            mapping="jordan_wigner",
            unit="angstrom",
        )
    else:
        # Fallback: keep the round-trip test meaningful even if cache_path was renamed.
        from common.paths import results_dir
        from common.naming import format_molecule_name

        path = results_dir("qpe") / f"{format_molecule_name('H2')}_{key}.json"

    atomic_write_json(path, payload)
    loaded = read_json(path)
    assert loaded == payload

    # If save_qpe_result/load_qpe_result exist, also exercise them (best-effort).
    if hasattr(qpe_io, "save_qpe_result") and hasattr(qpe_io, "load_qpe_result"):
        save_fn = getattr(qpe_io, "save_qpe_result")
        load_fn = getattr(qpe_io, "load_qpe_result")

        if callable(save_fn) and callable(load_fn):
            _call_flex(
                save_fn,
                result=payload,
                molecule="H2",
                n_ancilla=4,
                t=1.0,
                seed=0,
                shots=100,
                noise={"p_dep": 0.0, "p_amp": 0.0},
                trotter_steps=1,
                mapping="jordan_wigner",
                unit="angstrom",
                key=key,
            )
            out = _call_flex(
                load_fn,
                molecule="H2",
                n_ancilla=4,
                t=1.0,
                seed=0,
                shots=100,
                noise={"p_dep": 0.0, "p_amp": 0.0},
                trotter_steps=1,
                mapping="jordan_wigner",
                unit="angstrom",
                key=key,
            )
            # Some implementations return None if missing; at this point it should exist.
            assert out is not None
