from __future__ import annotations


def test_top_level_imports_smoke() -> None:
    import common
    import qpe
    import qite
    import vqe

    assert hasattr(vqe, "run_vqe")
    assert hasattr(qpe, "run_qpe")
    assert hasattr(qite, "run_qite")
    assert hasattr(common, "__file__")


def test_canonical_entrypoints_are_callable() -> None:
    from qpe import run_qpe
    from qite import run_qite
    from vqe import run_vqe

    assert callable(run_vqe)
    assert callable(run_qpe)
    assert callable(run_qite)


def test_versions_exist() -> None:
    import qpe
    import qite
    import vqe

    assert isinstance(vqe.__version__, str)
    assert isinstance(qpe.__version__, str)
    assert isinstance(qite.__version__, str)
