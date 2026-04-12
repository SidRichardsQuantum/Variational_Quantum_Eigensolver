from __future__ import annotations

import numpy as np
import pytest

from common.molecules import MOLECULES, get_molecule_config


def test_molecule_registry_integrity() -> None:
    for name, cfg in MOLECULES.items():
        assert isinstance(name, str)
        assert isinstance(cfg, dict)

        assert isinstance(cfg["symbols"], list)
        assert all(isinstance(s, str) for s in cfg["symbols"])

        coords = cfg["coordinates"]
        assert isinstance(coords, np.ndarray)
        assert coords.ndim == 2
        assert coords.shape[1] == 3

        assert isinstance(cfg["charge"], int)
        assert isinstance(cfg["basis"], str)
        assert len(cfg["symbols"]) == coords.shape[0]


def test_expected_molecules_exist() -> None:
    for name in [
        "H2",
        "H3+",
        "He2",
        "LiH",
        "H2O",
        "BeH2",
        "H4",
        "H5+",
        "H6",
        "HeH+",
    ]:
        assert name in MOLECULES


def test_get_molecule_config_returns_copy_like_dict() -> None:
    cfg = get_molecule_config("H2")
    assert isinstance(cfg, dict)
    assert cfg["symbols"] == ["H", "H"]
    assert cfg["coordinates"].shape[1] == 3


def test_unknown_molecule_raises() -> None:
    with pytest.raises(KeyError):
        get_molecule_config("NotAMolecule")
