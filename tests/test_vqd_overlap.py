"""Check VQD deflation values and gradients against one-qubit formulas."""

import numpy as np
import pennylane as qml
import pytest

from vqe.vqd import _state_overlap_metric


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("angle", [-0.7, 0.4, 0.6, 0.6 + np.pi])
def test_overlap_value_and_gradient(noisy, angle):
    dev = qml.device("default.mixed" if noisy else "default.qubit", wires=1)
    probability = 0.2
    reference_angle = 0.6

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def state(theta):
        qml.RX(theta, wires=0)
        if noisy:
            qml.DepolarizingChannel(probability, wires=0)
        return qml.state()

    reference = qml.numpy.array(state(reference_angle), requires_grad=False)
    theta = qml.numpy.array(angle, requires_grad=True)

    def overlap(theta):
        return _state_overlap_metric(reference, state(theta), noisy=noisy)

    # Depolarization contracts each state's Bloch vector by 1 - 4p/3.
    scale = (1 - 4 * probability / 3) ** 2 if noisy else 1.0
    delta = angle - reference_angle
    assert overlap(theta) == pytest.approx((1 + scale * np.cos(delta)) / 2)
    assert qml.grad(overlap)(theta) == pytest.approx(
        -scale * np.sin(delta) / 2, abs=1e-12
    )
