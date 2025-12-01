def test_molecule_registry_integrity():
    from vqe_qpe_common.molecules import MOLECULES
    import numpy as np

    for name, cfg in MOLECULES.items():
        assert isinstance(cfg["symbols"], list)
        assert all(isinstance(s, str) for s in cfg["symbols"])

        coords = cfg["coordinates"]
        assert isinstance(coords, np.ndarray)
        assert coords.shape[1] == 3  # xyz coords

        assert isinstance(cfg["charge"], int)
        assert isinstance(cfg["basis"], str)
