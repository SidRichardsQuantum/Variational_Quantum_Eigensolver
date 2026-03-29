from __future__ import annotations

import numpy as np
import pytest

from qite import run_qite


def test_qite_minimal_smoke() -> None:
    res = run_qite(
        molecule="H2",
        ansatz_name="UCCSD",
        steps=4,
        dtau=0.2,
        force=True,
        plot=False,
        show=False,
    )

    assert isinstance(res, dict)
    assert "energy" in res
    assert "energies" in res
    assert "num_qubits" in res
    assert "varqite" in res

    assert np.isfinite(float(res["energy"]))
    assert len(res["energies"]) >= 1
    assert int(res["num_qubits"]) > 0


def test_qite_rejects_noisy_optimization() -> None:
    with pytest.raises(ValueError):
        run_qite(
            molecule="H2",
            steps=2,
            dtau=0.2,
            noisy=True,
            force=True,
            plot=False,
            show=False,
        )
