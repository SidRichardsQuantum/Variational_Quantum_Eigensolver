"""
common.units
============

Shared unit helpers for coordinate inputs and geometry display.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_COORDINATE_UNITS = ("angstrom", "bohr")
ANGSTROM_PER_BOHR = 0.529177210903


def normalize_coordinate_unit(unit: str) -> str:
    unit_norm = str(unit).strip().lower()
    if unit_norm not in SUPPORTED_COORDINATE_UNITS:
        allowed = ", ".join(SUPPORTED_COORDINATE_UNITS)
        raise ValueError(
            f"Unsupported coordinate unit {unit!r}. Supported units: {allowed}."
        )
    return unit_norm


def convert_coordinates(coordinates, *, from_unit: str, to_unit: str) -> np.ndarray:
    from_norm = normalize_coordinate_unit(from_unit)
    to_norm = normalize_coordinate_unit(to_unit)
    coords = np.array(coordinates, dtype=float)

    if from_norm == to_norm:
        return coords

    if from_norm == "angstrom" and to_norm == "bohr":
        return coords / ANGSTROM_PER_BOHR
    if from_norm == "bohr" and to_norm == "angstrom":
        return coords * ANGSTROM_PER_BOHR

    raise AssertionError("unreachable")


def convert_length(value: float, *, from_unit: str, to_unit: str) -> float:
    return float(
        convert_coordinates(
            [[float(value), 0.0, 0.0]], from_unit=from_unit, to_unit=to_unit
        )[0, 0]
    )


def coordinate_unit_label(unit: str) -> str:
    unit_norm = normalize_coordinate_unit(unit)
    if unit_norm == "angstrom":
        return "Å"
    return "bohr"
