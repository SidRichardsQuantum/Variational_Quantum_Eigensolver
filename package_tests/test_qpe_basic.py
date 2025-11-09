import os
import pathlib
import pytest
import pennylane as qml
from pennylane import numpy as np

from qpe.core import run_qpe
from qpe.io_utils import (
    ensure_dirs,
    save_qpe_result,
    load_qpe_result,
    signature_hash,
)
from qpe.visualize import plot_qpe_distribution, IMG_DIR


@pytest.fixture(scope="module")
def simple_h2_hamiltonian():
    """Return a minimal H₂ molecular Hamiltonian and HF state."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.7414]])  # Å
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols, coordinates, charge=0, basis="STO-3G"
    )
    hf_state = qml.qchem.hf_state(2, n_qubits)
    return H, hf_state


def test_qpe_run_and_cache(simple_h2_hamiltonian):
    """Run a short QPE job and verify results are saved and reloaded from cache."""
    H, hf = simple_h2_hamiltonian
    ensure_dirs()

    # Use a fixed random seed for reproducibility
    np.random.seed(0)

    result = run_qpe(
        hamiltonian=H,
        hf_state=hf,
        n_ancilla=3,
        t=1.0,
        trotter_steps=1,
        shots=500,
        molecule_name="H2_test",
    )

    # Verify essential result fields
    for key in ["counts", "phase", "energy", "molecule"]:
        assert key in result, f"Missing key '{key}' in QPE result."

    # Save to cache
    save_qpe_result(result)

    # Recompute cache signature and confirm load works
    sig = signature_hash(
        molecule="H2_test",
        n_ancilla=3,
        t=1.0,
        noise=None,
        shots=500,
    )
    cached = load_qpe_result("H2_test", sig)

    assert cached is not None, "Cached QPE result not found."
    assert abs(result["energy"] - cached["energy"]) < 1e-8, "Cached energy mismatch."


def test_plot_qpe_distribution(simple_h2_hamiltonian):
    """Ensure QPE probability distribution plot is generated successfully (non-interactive)."""
    H, hf = simple_h2_hamiltonian
    ensure_dirs()

    result = run_qpe(
        hamiltonian=H,
        hf_state=hf,
        n_ancilla=3,
        t=1.0,
        trotter_steps=1,
        shots=500,
        molecule_name="H2_plot",
    )

    # Generate plot non-interactively (safe for CI)
    plot_qpe_distribution(result, show=False, save=True)

    # Ensure at least one PNG was created in qpe/images/
    images = list(pathlib.Path(IMG_DIR).glob("*.png"))
    assert len(images) >= 1, "No QPE plot images were generated."
