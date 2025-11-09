from vqe.ssvqe import run_ssvqe

def test_ssvqe_two_state_h2_rycz_smoke():
    res = run_ssvqe(
        molecule="H2",
        ansatz_name="RY-CZ",
        optimizer_name="GradientDescent",
        steps=5,
        penalty_weight=5.0,
        plot=False,
    )
    assert "energies_per_state" in res and len(res["energies_per_state"]) == 2
    assert len(res["energies_per_state"][0]) == 5
