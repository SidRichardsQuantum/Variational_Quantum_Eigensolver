# package_tests/test_reproducibility.py
from vqe.core import run_vqe


def test_h2_energy():
    """Check H₂ ground state energy (UCCSD, Adam) is close to reference."""
    result = run_vqe(
        molecule="H2",
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        n_steps=50,
        plot=False,
    )
    assert abs(result["energy"] + 1.136) < 0.05


def test_lih_energy():
    """Quick LiH energy sanity test (UCCSD, Adam)."""
    result = run_vqe(
        molecule="LiH",
        ansatz_name="UCCSD",
        optimizer_name="Adam",
        n_steps=30,
        plot=False,
        noisy=False,
    )
    assert result["energy"] < 0
