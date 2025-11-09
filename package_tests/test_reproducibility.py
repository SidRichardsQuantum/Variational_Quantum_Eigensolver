from vqe import run_vqe

def test_h2_energy():
    result = run_vqe("H2", ansatz_name="UCCSD", optimizer_name="Adam", n_steps=50, plot=False)
    assert abs(result["energy"] + 1.136) < 0.05

def test_lih_energy():
    """Fast LiH test mirroring LiH_Noiseless.ipynb (UCC doubles only)."""
    result = run_vqe(
        molecule="LiH",
        ansatz_name="UCCSD",
        optimizer_name="GradientDescent",
        n_steps=25,
        plot=False,
        shots=None,
        noise=False,
    )
    assert result["energy"] < 0
