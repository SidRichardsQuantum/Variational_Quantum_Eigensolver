def test_import_vqe_package():
    import vqe
    assert hasattr(vqe, "__file__")
