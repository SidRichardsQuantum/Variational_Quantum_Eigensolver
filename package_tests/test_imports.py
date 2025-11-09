def test_import_vqe_package():
    import vqe
    assert hasattr(vqe, "__file__")
    assert hasattr(vqe, "run_vqe")
    assert callable(vqe.run_vqe)


def test_import_qpe_package():
    import qpe
    assert hasattr(qpe, "__file__")
    assert hasattr(qpe, "run_qpe")
    assert callable(qpe.run_qpe)
