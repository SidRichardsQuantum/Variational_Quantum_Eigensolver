def test_package_imports():
    import vqe
    import qpe
    import vqe_qpe_common

    # Submodules
    import vqe.core
    import vqe.ansatz
    import vqe.engine
    import vqe.hamiltonian
    import vqe.optimizer

    import qpe.core
    import qpe.hamiltonian
    import qpe.noise

    import vqe_qpe_common.geometry
    import vqe_qpe_common.hamiltonian
    import vqe_qpe_common.molecules
    import vqe_qpe_common.plotting

    assert True  # If imports succeed, test passes
