"""Check noisy energy values and gradients independently of chemistry solvers."""

import numpy as np
import pennylane as qml
import pytest

from vqe.engine import make_energy_qnode


@pytest.mark.parametrize("diff_method", ["finite-diff", "backprop"])
@pytest.mark.parametrize("observable_kind", ["hamiltonian", "sum", "single"])
def test_noisy_energy_and_gradient(diff_method, observable_kind):
    probability = 0.1
    angle = qml.numpy.array(0.4, requires_grad=True)
    if observable_kind == "hamiltonian":
        observable = qml.Hamiltonian(
            [0.3, 0.7, -0.2], [qml.Identity(), qml.Z(0), qml.X(0)]
        )
    elif observable_kind == "sum":
        observable = 0.3 * qml.Identity() + 0.7 * qml.Z(0) - 0.2 * qml.X(0)
    else:
        observable = qml.Z(0)

    def ansatz(params, wires):
        qml.RY(params, wires=wires[0])

    energy = make_energy_qnode(
        observable,
        qml.device("default.mixed", wires=1),
        ansatz,
        1,
        noisy=True,
        depolarizing_prob=probability,
        diff_method=diff_method,
    )
    scale = 1 - 4 * probability / 3
    if observable_kind == "single":
        expected = scale * np.cos(angle)
        gradient = -scale * np.sin(angle)
    else:
        expected = 0.3 + scale * (0.7 * np.cos(angle) - 0.2 * np.sin(angle))
        gradient = scale * (-0.7 * np.sin(angle) - 0.2 * np.cos(angle))
    assert energy(angle) == pytest.approx(expected, abs=1e-10)
    assert qml.grad(energy)(angle) == pytest.approx(gradient, abs=1e-6)
