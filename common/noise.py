"""
common.noise
============

Shared built-in single-qubit noise channels used across VQE, QITE, and QPE.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pennylane as qml

BUILTIN_NOISE_FIELDS = (
    "p_dep",
    "p_amp",
    "p_phase_damp",
    "p_bit_flip",
    "p_phase_flip",
)


def apply_builtin_noise(
    wires: Iterable[int],
    *,
    depolarizing_prob: float = 0.0,
    amplitude_damping_prob: float = 0.0,
    phase_damping_prob: float = 0.0,
    bit_flip_prob: float = 0.0,
    phase_flip_prob: float = 0.0,
) -> None:
    """
    Apply supported single-qubit channels to every wire.
    """
    p_dep = float(depolarizing_prob)
    p_amp = float(amplitude_damping_prob)
    p_phase = float(phase_damping_prob)
    p_bit = float(bit_flip_prob)
    p_phase_flip = float(phase_flip_prob)

    if (
        (p_dep <= 0.0)
        and (p_amp <= 0.0)
        and (p_phase <= 0.0)
        and (p_bit <= 0.0)
        and (p_phase_flip <= 0.0)
    ):
        return

    for w in wires:
        if p_dep > 0.0:
            qml.DepolarizingChannel(p_dep, wires=w)
        if p_amp > 0.0:
            qml.AmplitudeDamping(p_amp, wires=w)
        if p_phase > 0.0:
            qml.PhaseDamping(p_phase, wires=w)
        if p_bit > 0.0:
            qml.BitFlip(p_bit, wires=w)
        if p_phase_flip > 0.0:
            qml.PhaseFlip(p_phase_flip, wires=w)


def format_noise_summary(noise: Mapping[str, object] | None) -> str:
    """
    Compact user-facing noise summary for titles / logs.
    """
    if not noise:
        return ""

    parts: list[str] = []
    mapping = (
        ("p_dep", "dep"),
        ("p_amp", "amp"),
        ("p_phase_damp", "phase"),
        ("p_bit_flip", "bit"),
        ("p_phase_flip", "phase_flip"),
    )
    for key, label in mapping:
        val = float(noise.get(key, 0.0) or 0.0)
        if val > 0.0:
            parts.append(f"{label}={val:g}")

    model = noise.get("model", None)
    if model not in {None, ""}:
        parts.append(f"model={model}")

    return ", ".join(parts)


def format_noise_tag(noise: Mapping[str, object] | None) -> str:
    """
    Filesystem-safe suffix for non-dep/amp built-in noise settings.
    """
    if not noise:
        return ""

    def _tok(val: float) -> str:
        return f"{val:g}".replace(".", "p")

    parts: list[str] = []
    mapping = (
        ("p_phase_damp", "phase"),
        ("p_bit_flip", "bit"),
        ("p_phase_flip", "phaseflip"),
    )
    for key, label in mapping:
        val = float(noise.get(key, 0.0) or 0.0)
        if val > 0.0:
            parts.append(f"{label}{_tok(val)}")

    return "_".join(parts)
