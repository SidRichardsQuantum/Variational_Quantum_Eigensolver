def test_package_imports():
    import vqe
    import qpe
    import common

    # Submodules
    import vqe.core
    import vqe.ansatz
    import vqe.engine
    import vqe.hamiltonian
    import vqe.optimizer

    import qpe.core
    import qpe.hamiltonian
    import qpe.noise

    import common.geometry
    import common.hamiltonian
    import common.molecules
    import common.plotting

    assert True  # If imports succeed, test passes
