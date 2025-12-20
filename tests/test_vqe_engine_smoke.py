def test_vqe_engine_runs_minimal():
    """
    Smoke test: ensure run_vqe executes at least 1 iteration.
    """
    from vqe import run_vqe

    result = run_vqe(
        molecule="H2",
        steps=2,
        stepsize=0.1,
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        noisy=False,
        force=True,
    )

    assert "energies" in result
    assert len(result["energies"]) >= 1
