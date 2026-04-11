from __future__ import annotations

import numpy as np

from common.molecules import get_molecule_config
from vqe import run_vqe
from vqe.ansatz import _build_ucc_data, init_params


def test_ucc_data_respects_charge_for_h3plus_geometry() -> None:
    cfg = get_molecule_config("H3+")

    singles_cation, doubles_cation, hf_cation = _build_ucc_data(
        cfg["symbols"],
        cfg["coordinates"],
        basis=cfg["basis"],
        charge=cfg["charge"],
    )
    singles_neutral, doubles_neutral, hf_neutral = _build_ucc_data(
        cfg["symbols"],
        cfg["coordinates"],
        basis=cfg["basis"],
        charge=0,
    )

    assert int(np.sum(hf_cation)) == 2
    assert int(np.sum(hf_neutral)) == 3
    assert not np.array_equal(hf_cation, hf_neutral)
    assert singles_cation != singles_neutral or doubles_cation != doubles_neutral


def test_ucc_init_params_respects_charge() -> None:
    cfg = get_molecule_config("H3+")
    num_wires = len(
        _build_ucc_data(
            cfg["symbols"],
            cfg["coordinates"],
            basis=cfg["basis"],
            charge=cfg["charge"],
        )[2]
    )

    charged_params = init_params(
        ansatz_name="UCCSD",
        num_wires=num_wires,
        symbols=cfg["symbols"],
        coordinates=cfg["coordinates"],
        basis=cfg["basis"],
        charge=cfg["charge"],
        seed=0,
    )
    neutral_params = init_params(
        ansatz_name="UCCSD",
        num_wires=num_wires,
        symbols=cfg["symbols"],
        coordinates=cfg["coordinates"],
        basis=cfg["basis"],
        charge=0,
        seed=0,
    )

    charged_singles, charged_doubles, _ = _build_ucc_data(
        cfg["symbols"],
        cfg["coordinates"],
        basis=cfg["basis"],
        charge=cfg["charge"],
    )
    neutral_singles, neutral_doubles, _ = _build_ucc_data(
        cfg["symbols"],
        cfg["coordinates"],
        basis=cfg["basis"],
        charge=0,
    )

    assert len(charged_params) == len(charged_singles) + len(charged_doubles)
    assert len(neutral_params) == len(neutral_singles) + len(neutral_doubles)
    assert np.allclose(np.asarray(charged_params, dtype=float), 0.0)
    assert np.allclose(np.asarray(neutral_params, dtype=float), 0.0)


def test_charged_uccsd_vqe_smoke() -> None:
    res = run_vqe(
        molecule="H3+",
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        steps=1,
        stepsize=0.1,
        noisy=False,
        force=True,
        plot=False,
    )

    assert isinstance(res, dict)
    assert np.isfinite(float(res["energy"]))
    assert len(res["final_params"]) >= 1
