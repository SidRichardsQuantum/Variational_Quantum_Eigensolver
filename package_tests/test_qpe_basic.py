import os
import json
import pytest
import pennylane as qml
from pennylane import numpy as np
from qpe.core import run_qpe
from qpe.io_utils import ensure_dirs, save_qpe_result, load_qpe_result
from qpe.visualize import plot_qpe_distribution


@pytest.fixture(scope="module")
def simple_h2_hamiltonian():
    """Return a minimal H₂ molecular Hamiltonian and HF state."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.7414]])  # Å
    H, n_qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=0, basis="STO-3G")
    hf_state = qml.qchem.hf_state(2, n_qubits)
    return H, hf_state


def test_qpe_run_and_cache(simple_h2_hamiltonian):
    """Run a short QPE job and verify that results are cached."""
    H, hf = simple_h2_hamiltonian
    ensure_dirs()

    result = run_qpe(
        hamiltonian=H,
        hf_state=hf,
        n_ancilla=3,
        t=0.5,
        trotter_steps=1,
        shots=200,
        molecule_name="H2_test",
    )

    # Basic keys
    for key in ["counts", "phase", "energy", "molecule"]:
        assert key in result

    # Save and reload cache
    save_qpe_result(result)
    import hashlib, json
    sig = hashlib.md5(json.dumps({
        "mol": "H2_test", "anc": 3, "t": 0.5, "noise": None, "shots": 200
    }, sort_keys=True).encode()).hexdigest()
    cached = load_qpe_result("H2_test", sig)
    assert cached is not None
    assert abs(result["energy"] - cached["energy"]) < 1e-8


def test_plot_qpe_distribution(tmp_path, simple_h2_hamiltonian):
    """Ensure QPE distribution plot is generated without errors."""
    import pathlib
    from qpe.visualize import IMG_DIR

    H, hf = simple_h2_hamiltonian
    ensure_dirs()

    result = run_qpe(
        hamiltonian=H,
        hf_state=hf,
        n_ancilla=3,
        t=0.5,
        trotter_steps=1,
        shots=100,
        molecule_name="H2_plot",
    )

    plot_qpe_distribution(result)

    images = list(pathlib.Path(IMG_DIR).glob("*.png"))
    assert len(images) >= 1
