from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import qite.__main__ as qite_main
import vqe.core as vqe_core


def test_vqe_optimizer_noise_stats_supports_phase_damping(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_run_vqe(**kwargs):
        calls.append(kwargs)
        return {
            "energy": -1.0,
            "final_state_real": [1.0, 0.0],
            "final_state_imag": [0.0, 0.0],
        }

    monkeypatch.setattr(vqe_core, "run_vqe", fake_run_vqe)

    out = vqe_core.run_vqe_optimizer_comparison(
        molecule="H2",
        ansatz_name="UCCSD",
        optimizers=["Adam"],
        steps=2,
        stepsize=0.1,
        mode="noise_stats",
        noise_type="phase_damping",
        noise_levels=[0.0, 0.1],
        seeds=[0],
        plot=False,
        show=False,
    )

    noisy_calls = [c for c in calls if c["noisy"]]
    assert out["noise_type"] == "phase_damping"
    assert [float(c["phase_damping_prob"]) for c in noisy_calls] == [0.0, 0.1]
    assert all(float(c["depolarizing_prob"]) == 0.0 for c in noisy_calls)
    assert all(float(c["amplitude_damping_prob"]) == 0.0 for c in noisy_calls)
    assert all(float(c["bit_flip_prob"]) == 0.0 for c in noisy_calls)
    assert all(float(c["phase_flip_prob"]) == 0.0 for c in noisy_calls)


def test_qite_eval_noise_sweeps_selected_channel(monkeypatch) -> None:
    calls: list[dict[str, float]] = []

    monkeypatch.setattr(
        qite_main,
        "_unpack_hamiltonian_metadata",
        lambda **_kwargs: (
            "H",
            2,
            np.array([1, 0], dtype=int),
            ["H", "H"],
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], dtype=float),
            "sto-3g",
            0,
            "jordan_wigner",
            "angstrom",
        ),
    )
    monkeypatch.setattr(
        qite_main,
        "run_qite",
        lambda **_kwargs: {
            "energy": -1.0,
            "final_params_shape": [1],
            "final_params": [0.0],
        },
    )

    def fake_noisy_eval_energy_and_diag(**kwargs):
        calls.append(
            {
                "dep": float(kwargs["dep"]),
                "amp": float(kwargs["amp"]),
                "phase": float(kwargs["phase"]),
                "bit": float(kwargs["bit"]),
                "phase_flip": float(kwargs["phase_flip"]),
            }
        )
        return -1.0, np.array([1.0, 0.0])

    monkeypatch.setattr(
        qite_main, "_noisy_eval_energy_and_diag", fake_noisy_eval_energy_and_diag
    )

    args = SimpleNamespace(
        molecule="H2",
        ansatz="UCCSD",
        steps=2,
        dtau=0.1,
        seed=0,
        basis="sto-3g",
        charge=0,
        symbols=None,
        coordinates=None,
        mapping="jordan_wigner",
        unit="angstrom",
        depolarizing_prob=0.02,
        amplitude_damping_prob=0.03,
        phase_damping_prob=0.0,
        bit_flip_prob=0.0,
        phase_flip_prob=0.0,
        sweep_noise_type="phase_damping",
        sweep_levels="0.00,0.10",
        seeds="0",
        force=False,
        fd_eps=1e-3,
        reg=1e-6,
        solver="solve",
        pinv_rcond=1e-10,
    )

    out = qite_main.eval_noise(args)

    assert out["mode"] == "sweep_noise"
    assert out["sweep_noise_type"] == "phase_damping"
    assert out["sweep_levels"] == [0.0, 0.1]
    assert calls == [
        {"dep": 0.02, "amp": 0.03, "phase": 0.0, "bit": 0.0, "phase_flip": 0.0},
        {"dep": 0.02, "amp": 0.03, "phase": 0.1, "bit": 0.0, "phase_flip": 0.0},
    ]
