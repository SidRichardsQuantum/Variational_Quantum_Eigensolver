"""Check VQD deflation values and gradients against one-qubit formulas."""

import numpy as np
import pennylane as qml
import pytest

from vqe.vqd import _state_overlap_metric
from vqe import run_vqd


@pytest.mark.filterwarnings("error::numpy.exceptions.ComplexWarning")
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


@pytest.mark.filterwarnings("error::numpy.exceptions.ComplexWarning")
@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("mapping", ["jordan_wigner", "parity", "bravyi_kitaev"])
def test_ucc_deflation_optimization_and_cache(noisy, mapping, monkeypatch):
    kwargs = dict(
        molecule="H2",
        ansatz_name="UCCSD",
        mapping=mapping,
        num_states=2,
        steps=2,
        beta_start=2.0,
        beta=2.0,
        noisy=noisy,
        depolarizing_prob=0.1 if noisy else 0.0,
        plot=False,
    )
    result = run_vqd(**kwargs)
    energies = np.asarray(result["energies_per_state"])
    assert energies.shape == (2, 2)
    assert np.all(np.isfinite(energies))
    assert np.all(np.isfinite(result["final_params"]))
    assert result["config"]["state_diff_method"] == "backprop"

    from vqe.io_utils import run_signature

    legacy_config = dict(result["config"])
    legacy_config.pop("state_diff_method")
    assert run_signature(legacy_config) != run_signature(result["config"])

    def unexpected_optimizer(*args, **kwargs):
        pytest.fail("A cache hit must not optimize again")

    monkeypatch.setattr("vqe.vqd.build_optimizer", unexpected_optimizer)
    assert run_vqd(**kwargs) == result
