import pytest


def test_molecule_registry_has_new_entries():
    """Ensure newly added molecules exist in the central registry."""
    from vqe_qpe_common.molecules import MOLECULES

    expected = ["BeH2", "H4", "HeH+"]
    for name in expected:
        assert name in MOLECULES, f"Molecule '{name}' missing from registry."


def test_get_molecule_config_returns_dict():
    """Ensure get_molecule_config works for new molecules."""
    from vqe_qpe_common.molecules import get_molecule_config

    for name in ["BeH2", "H4", "HeH+"]:
        cfg = get_molecule_config(name)
        assert isinstance(cfg, dict)
        assert "symbols" in cfg
        assert "coordinates" in cfg
        assert "charge" in cfg
        assert "basis" in cfg


def test_unknown_molecule_raises():
    """Ensure error is raised for unknown molecule names."""
    from vqe_qpe_common.molecules import get_molecule_config

    with pytest.raises(KeyError):
        get_molecule_config("NotAMolecule")
