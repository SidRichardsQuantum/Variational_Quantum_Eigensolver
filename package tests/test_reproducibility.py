from vqe import run_vqe

def test_h2_energy():
    result = run_vqe("H2", ansatz_name="UCCSD", optimizer_name="Adam", n_steps=10, plot=False)
    assert abs(result["energy"] + 1.136) < 0.05

def test_lih_energy():
    result = run_vqe("LiH", ansatz_name="UCCSD", optimizer_name="Adam", n_steps=10, plot=False)
    assert result["energy"] < 0
