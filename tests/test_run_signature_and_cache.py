from __future__ import annotations

import copy

import numpy as np

from common.persist import atomic_write_json, read_json
from qite.io_utils import (
    load_run_record as qite_load_run_record,
    make_filename_prefix as qite_make_filename_prefix,
    make_run_config_dict as qite_make_run_config_dict,
    run_signature as qite_run_signature,
    save_run_record as qite_save_run_record,
)
from qpe.io_utils import cache_path as qpe_cache_path
from qpe.io_utils import signature_hash as qpe_signature_hash
from vqe.io_utils import (
    load_run_record as vqe_load_run_record,
    make_filename_prefix as vqe_make_filename_prefix,
    make_run_config_dict as vqe_make_run_config_dict,
    run_signature as vqe_run_signature,
    save_run_record as vqe_save_run_record,
)


def test_vqe_run_signature_stable_across_semantic_equivalents() -> None:
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
            "float_py": 0.30000000000000004,
            "float_np": np.float64(0.3),
            "int_np": np.int64(7),
        },
    }

    sig1 = vqe_run_signature(cfg)

    cfg2 = copy.deepcopy(cfg)
    cfg2["geometry"] = cfg["geometry"].tolist()
    cfg2["meta"]["float_py"] = 0.3
    cfg2["meta"]["float_np"] = 0.30000000000000004
    cfg2["meta"]["int_np"] = int(cfg2["meta"]["int_np"])

    sig2 = vqe_run_signature(cfg2)
    assert sig1 == sig2


def test_qite_run_signature_stable_across_semantic_equivalents() -> None:
    cfg = {
        "molecule": "H2",
        "symbols": ["H", "H"],
        "coordinates": np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
        "basis": "sto-3g",
        "charge": 0,
        "unit": "angstrom",
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
    }

    sig1 = qite_run_signature(cfg)

    cfg2 = copy.deepcopy(cfg)
    cfg2["coordinates"] = cfg["coordinates"].tolist()
    cfg2["varqite"]["fd_eps"] = float(cfg2["varqite"]["fd_eps"])

    sig2 = qite_run_signature(cfg2)
    assert sig1 == sig2


def test_vqe_cache_roundtrip() -> None:
    cfg = vqe_make_run_config_dict(
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

    h = vqe_run_signature(cfg)
    prefix = vqe_make_filename_prefix(cfg, noisy=False, seed=0, hash_str=h, algo="vqe")

    record = {
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

    vqe_save_run_record(prefix, record)
    loaded = vqe_load_run_record(prefix)

    assert loaded is not None
    assert loaded == record


def test_vqe_run_signature_normalizes_alias_names() -> None:
    adam_canonical = vqe_make_run_config_dict(
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
    gradient_alias = vqe_make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        ansatz_desc="ucc sd",
        optimizer_name="gradient descent",
        stepsize=0.2,
        max_iterations=5,
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        molecule_label="H2",
    )

    gradient_canonical = vqe_make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        ansatz_desc="UCCSD",
        optimizer_name="GradientDescent",
        stepsize=0.2,
        max_iterations=5,
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        molecule_label="H2",
    )

    adam_alias = vqe_make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        ansatz_desc="UCC-SD",
        optimizer_name="adam",
        stepsize=0.2,
        max_iterations=5,
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        molecule_label="H2",
    )

    assert adam_canonical["ansatz"] == gradient_alias["ansatz"] == "UCCSD"
    assert gradient_canonical["ansatz"] == "UCCSD"
    assert (
        gradient_canonical["optimizer"]["name"] == gradient_alias["optimizer"]["name"]
    )
    assert gradient_alias["optimizer"]["name"] == "GradientDescent"
    assert (
        adam_canonical["optimizer"]["name"] == adam_alias["optimizer"]["name"] == "Adam"
    )
    assert vqe_run_signature(gradient_canonical) == vqe_run_signature(gradient_alias)
    assert vqe_run_signature(adam_canonical) == vqe_run_signature(adam_alias)


def test_qite_cache_roundtrip() -> None:
    cfg = qite_make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        unit="angstrom",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        dtau=0.1,
        steps=10,
        molecule_label="H2",
        ansatz_name="UCCSD",
    )

    h = qite_run_signature(cfg)
    prefix = qite_make_filename_prefix(
        cfg, noisy=False, seed=0, hash_str=h, algo="qite"
    )

    record = {
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

    qite_save_run_record(prefix, record)
    loaded = qite_load_run_record(prefix)

    assert loaded is not None
    assert loaded == record


def test_qite_run_signature_normalizes_ansatz_alias_names() -> None:
    canonical = qite_make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        unit="angstrom",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        dtau=0.2,
        steps=75,
        molecule_label="H2",
        ansatz_name="UCCSD",
    )
    alias = qite_make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        unit="angstrom",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        dtau=0.2,
        steps=75,
        molecule_label="H2",
        ansatz_name="ucc sd",
    )

    assert canonical["ansatz"] == alias["ansatz"] == "UCCSD"
    assert qite_run_signature(canonical) == qite_run_signature(alias)


def test_qpe_cache_roundtrip() -> None:
    key = qpe_signature_hash(
        molecule="H2",
        symbols=["H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1, "p_amp": 0.0},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )

    path = qpe_cache_path(
        molecule="H2",
        n_ancilla=4,
        t=1.0,
        seed=0,
        noise={"p_dep": 0.1, "p_amp": 0.0},
        key=key,
        mapping="jordan_wigner",
        unit="angstrom",
    )

    payload = {
        "config": {"molecule": "H2", "n_ancilla": 4},
        "result": {"phase": 0.25, "energy": -1.0},
    }

    atomic_write_json(path, payload)
    loaded = read_json(path)

    assert loaded == payload


def test_qpe_signature_hash_changes_with_active_space() -> None:
    base = qpe_signature_hash(
        molecule="LiH",
        symbols=["Li", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
        basis="sto-3g",
        charge=0,
        n_ancilla=2,
        t=1.0,
        seed=0,
        shots=100,
        noise={},
        trotter_steps=1,
        mapping="jordan_wigner",
        unit="angstrom",
        active_electrons=2,
        active_orbitals=2,
    )
    changed = qpe_signature_hash(
        molecule="LiH",
        symbols=["Li", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
        basis="sto-3g",
        charge=0,
        n_ancilla=2,
        t=1.0,
        seed=0,
        shots=100,
        noise={},
        trotter_steps=1,
        mapping="jordan_wigner",
        unit="angstrom",
        active_electrons=2,
        active_orbitals=3,
    )

    assert base != changed


def test_qite_run_signature_changes_with_charge_and_unit() -> None:
    cfg = qite_make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        unit="angstrom",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        dtau=0.1,
        steps=10,
        molecule_label="H2",
        ansatz_name="UCCSD",
    )
    cfg_charge = copy.deepcopy(cfg)
    cfg_charge["charge"] = 1

    cfg_unit = copy.deepcopy(cfg)
    cfg_unit["unit"] = "bohr"

    assert qite_run_signature(cfg) != qite_run_signature(cfg_charge)
    assert qite_run_signature(cfg) != qite_run_signature(cfg_unit)


def test_vqe_run_signature_changes_with_active_space() -> None:
    cfg = vqe_make_run_config_dict(
        symbols=["Li", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
        basis="sto-3g",
        ansatz_desc="UCCSD",
        optimizer_name="Adam",
        stepsize=0.2,
        max_iterations=5,
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        molecule_label="LiH",
        charge=0,
        unit="angstrom",
        active_electrons=2,
        active_orbitals=2,
    )
    cfg2 = copy.deepcopy(cfg)
    cfg2["active_orbitals"] = 3

    assert vqe_run_signature(cfg) != vqe_run_signature(cfg2)


def test_qite_run_signature_changes_with_active_space() -> None:
    cfg = qite_make_run_config_dict(
        symbols=["Li", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]],
        basis="sto-3g",
        charge=0,
        unit="angstrom",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        dtau=0.1,
        steps=10,
        molecule_label="LiH",
        ansatz_name="UCCSD",
        active_electrons=2,
        active_orbitals=2,
    )
    cfg2 = copy.deepcopy(cfg)
    cfg2["active_electrons"] = 4

    assert qite_run_signature(cfg) != qite_run_signature(cfg2)


def test_qpe_signature_changes_with_geometry_metadata() -> None:
    base = qpe_signature_hash(
        molecule="custom",
        symbols=["H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1, "p_amp": 0.0},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )
    changed_geometry = qpe_signature_hash(
        molecule="custom",
        symbols=["H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.20]],
        basis="sto-3g",
        charge=0,
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1, "p_amp": 0.0},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )
    changed_charge = qpe_signature_hash(
        molecule="custom",
        symbols=["H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=1,
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1, "p_amp": 0.0},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )

    assert base != changed_geometry
    assert base != changed_charge


def test_qpe_signature_changes_with_extended_noise_fields() -> None:
    base = qpe_signature_hash(
        molecule="H2",
        symbols=["H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )
    changed_phase = qpe_signature_hash(
        molecule="H2",
        symbols=["H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1, "p_phase_damp": 0.02},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )
    changed_bit = qpe_signature_hash(
        molecule="H2",
        symbols=["H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1, "p_bit_flip": 0.02},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )
    changed_phase_flip = qpe_signature_hash(
        molecule="H2",
        symbols=["H", "H"],
        geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        charge=0,
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1, "p_phase_flip": 0.02},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )

    assert base != changed_phase
    assert base != changed_bit
    assert base != changed_phase_flip
