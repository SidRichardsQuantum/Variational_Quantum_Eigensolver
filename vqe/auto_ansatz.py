"""Automatic ansatz selection for expert-mode qubit Hamiltonians."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pennylane as qml

from vqe.ansatz import canonicalize_ansatz_name

_AUTO_DEFAULTS = {
    "TFIM-HVA": {"layers": 4},
    "XXZ-HVA": {"layers": 4},
    "NumberPreservingGivens": {"layers": 3},
    "StronglyEntanglingLayers": {"layers": 2},
}


def _is_auto_name(ansatz_name: str) -> bool:
    return str(ansatz_name).strip().lower() in {"auto", "automatic"}


def _term_signature(pauli_word) -> tuple[tuple[int, str], ...]:
    return tuple(sorted((int(wire), str(op)) for wire, op in pauli_word.items()))


def _pauli_terms(hamiltonian) -> list[tuple[tuple[tuple[int, str], ...], complex]]:
    sentence = qml.pauli.pauli_sentence(hamiltonian)
    return [
        (_term_signature(word), complex(coeff))
        for word, coeff in sentence.items()
        if abs(complex(coeff)) > 1e-12
    ]


def _is_nearest_neighbour_chain(pairs: set[tuple[int, int]]) -> bool:
    return bool(pairs) and all(abs(left - right) == 1 for left, right in pairs)


def _detect_ansatz(hamiltonian, num_qubits: int) -> tuple[str, str]:
    try:
        terms = _pauli_terms(hamiltonian)
    except Exception as exc:
        return (
            "StronglyEntanglingLayers",
            f"could not inspect Hamiltonian Pauli terms ({type(exc).__name__})",
        )

    if not terms:
        return "StronglyEntanglingLayers", "Hamiltonian has no non-zero Pauli terms"

    one_body_axes: set[str] = set()
    two_body_axes_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    two_body_coeffs: dict[tuple[tuple[int, int], str], complex] = defaultdict(complex)
    unsupported_terms: list[tuple[tuple[int, str], ...]] = []

    for word, coeff in terms:
        if len(word) == 0:
            continue
        if len(word) == 1:
            one_body_axes.add(word[0][1])
        elif len(word) == 2:
            (left, axis_left), (right, axis_right) = word
            if axis_left != axis_right:
                unsupported_terms.append(word)
                continue
            pair = tuple(sorted((left, right)))
            two_body_axes_by_pair[pair].add(axis_left)
            two_body_coeffs[(pair, axis_left)] += coeff
        else:
            unsupported_terms.append(word)

    if unsupported_terms:
        return (
            "StronglyEntanglingLayers",
            "found mixed-axis or higher-body Pauli terms",
        )

    pairs = set(two_body_axes_by_pair)
    nearest_neighbour = _is_nearest_neighbour_chain(pairs)

    if (
        nearest_neighbour
        and one_body_axes.issubset({"X"})
        and pairs
        and all(axes == {"Z"} for axes in two_body_axes_by_pair.values())
    ):
        return "TFIM-HVA", "detected nearest-neighbour ZZ couplings with X fields"

    if nearest_neighbour and not one_body_axes and pairs:
        pair_axes = list(two_body_axes_by_pair.values())
        has_xy_pairs = all({"X", "Y"}.issubset(axes) for axes in pair_axes)
        has_z = any("Z" in axes for axes in pair_axes)
        only_xy_or_z = all(axes.issubset({"X", "Y", "Z"}) for axes in pair_axes)
        xy_coeffs_match = all(
            abs(two_body_coeffs[(pair, "X")] - two_body_coeffs[(pair, "Y")]) < 1e-8
            for pair, axes in two_body_axes_by_pair.items()
            if {"X", "Y"}.issubset(axes)
        )

        if has_xy_pairs and only_xy_or_z and xy_coeffs_match:
            if has_z:
                return (
                    "XXZ-HVA",
                    "detected nearest-neighbour XX+YY exchange with ZZ couplings",
                )
            return (
                "NumberPreservingGivens",
                "detected nearest-neighbour XX+YY exchange without ZZ couplings",
            )

    return (
        "StronglyEntanglingLayers",
        f"no confident model match for {int(num_qubits)}-qubit Pauli structure",
    )


def resolve_auto_ansatz(
    ansatz_name: str,
    hamiltonian,
    num_qubits: int,
    ansatz_kwargs: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Resolve ``ansatz_name='auto'`` into a concrete ansatz and kwargs."""
    requested = str(ansatz_name)
    if not _is_auto_name(requested):
        return canonicalize_ansatz_name(requested), dict(ansatz_kwargs or {}), None

    selected, reason = _detect_ansatz(hamiltonian, int(num_qubits))
    resolved_kwargs = dict(_AUTO_DEFAULTS.get(selected, {}))
    resolved_kwargs.update(dict(ansatz_kwargs or {}))
    selection = {
        "requested": requested,
        "selected": selected,
        "reason": reason,
        "ansatz_kwargs": dict(resolved_kwargs),
    }
    return selected, resolved_kwargs, selection
