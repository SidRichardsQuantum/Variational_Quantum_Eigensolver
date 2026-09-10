"""Reversible occupation-number encodings for chemistry circuits."""

from functools import lru_cache

import numpy as np
import pennylane as qml


@lru_cache(maxsize=None)
def encoding_cnots(num_wires: int, mapping: str) -> tuple[tuple[int, int], ...]:
    """CNOT network taking occupation bits to the requested qubit encoding."""
    if mapping == "jordan_wigner":
        return ()
    if mapping == "parity":
        return tuple((i, i + 1) for i in range(num_wires - 1))
    if mapping == "bravyi_kitaev":
        return tuple(
            (i, i | (i + 1)) for i in range(num_wires) if (i | (i + 1)) < num_wires
        )
    raise ValueError("mapping must be 'jordan_wigner', 'parity', or 'bravyi_kitaev'.")


def apply_encoding(wires, mapping: str) -> None:
    """Encode a circuit prepared in the occupation-number basis."""
    wires = list(wires)
    for control, target in encoding_cnots(len(wires), mapping):
        qml.CNOT(wires=[wires[control], wires[target]])


def occupation_bits(bits, mapping: str) -> np.ndarray:
    """Decode a mapped computational-basis reference for a chemistry ansatz."""
    out = np.array(bits, dtype=int, copy=True)
    for control, target in reversed(encoding_cnots(len(out), mapping)):
        out[target] ^= out[control]
    return out
