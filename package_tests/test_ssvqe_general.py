# package_tests/test_ssvqe_general.py
from vqe.ssvqe import run_ssvqe


def test_ssvqe_two_state_h2_rycz_smoke():
    """Smoke test: 2-state SSVQE on H2 with RY-CZ ansatz and GradientDescent optimizer."""
    res = run_ssvqe(
        molecule="H2",
        ansatz_name="RY-CZ",
        optimizer_name="GradientDescent",
        steps=50,
        penalty_weight=10.0,
        noisy=False,
        plot=False,
    )

    # Structural sanity checks
    assert isinstance(res, dict)
    assert "energies_per_state" in res
    assert len(res["energies_per_state"]) == 2
    assert len(res["energies_per_state"][0]) == 5

    # Numerical sanity check (energies should be finite)
    for state_energies in res["energies_per_state"]:
        assert all(abs(e) < 10 for e in state_energies), "Energies diverged or invalid"
