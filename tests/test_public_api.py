from __future__ import annotations


def test_top_level_imports_smoke() -> None:
    import common
    import qpe
    import qite
    import vqe

    assert hasattr(vqe, "run_vqe")
    assert hasattr(qpe, "run_qpe")
    assert hasattr(qite, "run_qite")
    assert hasattr(qite, "run_qrte")
    assert hasattr(common, "__file__")


def test_canonical_entrypoints_are_callable() -> None:
    from qpe import run_qpe
    from qite import run_qite, run_qrte
    from vqe import run_vqe

    assert callable(run_vqe)
    assert callable(run_qpe)
    assert callable(run_qite)
    assert callable(run_qrte)


def test_versions_exist() -> None:
    import qpe
    import qite
    import vqe

    assert isinstance(vqe.__version__, str)
    assert isinstance(qpe.__version__, str)
    assert isinstance(qite.__version__, str)


def test_calibrated_main_defaults() -> None:
    import inspect

    from qpe import run_qpe
    from qite import run_qite
    from vqe import run_vqe

    vqe_sig = inspect.signature(run_vqe)
    qite_sig = inspect.signature(run_qite)
    qpe_sig = inspect.signature(run_qpe)

    assert vqe_sig.parameters["ansatz_name"].default == "UCCSD"
    assert vqe_sig.parameters["optimizer_name"].default == "Adam"
    assert vqe_sig.parameters["stepsize"].default == 0.2
    assert vqe_sig.parameters["steps"].default == 75

    assert qite_sig.parameters["ansatz_name"].default == "UCCSD"
    assert qite_sig.parameters["dtau"].default == 0.2
    assert qite_sig.parameters["steps"].default == 75

    assert qpe_sig.parameters["n_ancilla"].default == 4
    assert qpe_sig.parameters["t"].default == 1.0
    assert qpe_sig.parameters["trotter_steps"].default == 2
    assert qpe_sig.parameters["shots"].default == 1000


def test_ansatz_registry_exposes_only_canonical_ucc_names() -> None:
    from vqe.ansatz import ANSATZES, get_ansatz

    assert "UCCSD" in ANSATZES
    assert "UCCD" in ANSATZES
    assert "UCCS" in ANSATZES
    assert "UCC-SD" not in ANSATZES
    assert "UCC-D" not in ANSATZES
    assert "UCC-S" not in ANSATZES

    assert get_ansatz("UCC-SD") is get_ansatz("UCCSD")
    assert get_ansatz("UCC-D") is get_ansatz("UCCD")
    assert get_ansatz("UCC-S") is get_ansatz("UCCS")
    assert get_ansatz("ucc sd") is get_ansatz("UCCSD")
    assert get_ansatz("ucc_d") is get_ansatz("UCCD")
    assert get_ansatz("strongly entangling layers") is get_ansatz(
        "StronglyEntanglingLayers"
    )


def test_optimizer_registry_exposes_only_canonical_names() -> None:
    from vqe.optimizer import OPTIMIZERS, get_optimizer

    assert "Adam" in OPTIMIZERS
    assert "GradientDescent" in OPTIMIZERS
    assert "NesterovMomentum" in OPTIMIZERS
    assert "adam" not in OPTIMIZERS
    assert "gd" not in OPTIMIZERS
    assert "Nesterov" not in OPTIMIZERS

    assert type(get_optimizer("adam")).__name__ == type(get_optimizer("Adam")).__name__
    assert (
        type(get_optimizer("gd")).__name__
        == type(get_optimizer("GradientDescent")).__name__
    )
    assert (
        type(get_optimizer("Nesterov")).__name__
        == type(get_optimizer("NesterovMomentum")).__name__
    )
    assert (
        type(get_optimizer("gradient descent")).__name__
        == type(get_optimizer("GradientDescent")).__name__
    )
    assert (
        type(get_optimizer("rms prop")).__name__
        == type(get_optimizer("RMSProp")).__name__
    )
