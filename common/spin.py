"""Spin occupations and excitation lists in interleaved alpha/beta order."""

from itertools import combinations

import numpy as np

from common.encoding import occupation_bits


def reference_occupation(
    electrons: int, orbitals: int, multiplicity: int
) -> np.ndarray:
    """Prepare the highest-weight determinant, with M_s = (multiplicity - 1)/2."""
    unpaired = multiplicity - 1
    if (
        electrons < 0
        or orbitals < 0
        or orbitals % 2
        or unpaired < 0
        or unpaired > electrons
        or (electrons - unpaired) % 2
        or electrons + unpaired > orbitals
    ):
        raise ValueError(
            "Electron count, orbital count, and multiplicity are incompatible."
        )
    n_alpha = (electrons + unpaired) // 2
    n_beta = (electrons - unpaired) // 2
    bits = np.zeros(orbitals, dtype=int)
    bits[: 2 * n_alpha : 2] = 1
    bits[1 : 2 * n_beta : 2] = 1
    return bits


def reference_multiplicity(bits, mapping: str = "jordan_wigner") -> int:
    """Recover the multiplicity of a highest-weight chemistry reference."""
    occupied = occupation_bits(bits, mapping)
    return int(abs(sum(occupied[::2]) - sum(occupied[1::2])) + 1)


def reference_excitations(bits) -> tuple[list[tuple], list[tuple]]:
    """Occupied-to-virtual singles/doubles conserving particle number and M_s.

    Conserving M_s alone does not guarantee an eigenstate of total spin S^2.
    """
    occupied = [i for i, bit in enumerate(bits) if bit]
    virtual = [i for i, bit in enumerate(bits) if not bit]
    singles = [(i, a) for i in occupied for a in virtual if i % 2 == a % 2]
    doubles = [
        (i, j, a, b)
        for i, j in combinations(occupied, 2)
        for a, b in combinations(virtual, 2)
        if i % 2 + j % 2 == a % 2 + b % 2
    ]
    return singles, doubles
