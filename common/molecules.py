"""
common.molecules
================

Canonical molecule registry shared by VQE and QPE.

Every molecule entry contains:
    • symbols      (list[str])
    • coordinates  (np.ndarray)
    • charge       (int)
    • basis        (str)
    • unit         (stored coordinate unit; registry values use angstrom)
"""

from __future__ import annotations

import numpy as np

MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.7414],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    "H3+": {
        "symbols": ["H", "H", "H"],
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.872],
                [0.755, 0.0, 0.436],
            ]
        ),
        "charge": +1,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    "He2": {
        "symbols": ["He", "He"],
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.50],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.6],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    "H2O": {
        "symbols": ["O", "H", "H"],
        "coordinates": np.array(
            [
                [0.000000, 0.000000, 0.000000],
                [0.758602, 0.000000, 0.504284],
                [-0.758602, 0.000000, 0.504284],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    # ------------------------------------------------------
    # NEW MOLECULES (BeH2, H4-chain, HeH+)
    # ------------------------------------------------------
    "HeH+": {
        "symbols": ["He", "H"],
        # Typical HeH+ bond length ~1.46 Å
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.46],
            ]
        ),
        "charge": +1,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    "BeH2": {
        "symbols": ["Be", "H", "H"],
        # Linear geometry: H–Be–H with ~1.33 Å Be–H bond length
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.33],
                [0.0, 0.0, -1.33],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    "H4": {
        "symbols": ["H", "H", "H", "H"],
        # Linear H4 chain, equally spaced at 1.0 Å
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    "H5+": {
        "symbols": ["H", "H", "H", "H", "H"],
        # Linear H5+ chain, equally spaced at 1.0 Å
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
            ]
        ),
        "charge": +1,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
    "H6": {
        "symbols": ["H", "H", "H", "H", "H", "H"],
        # Linear H6 chain, equally spaced at 1.0 Å
        "coordinates": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        ),
        "charge": 0,
        "basis": "STO-3G",
        "unit": "angstrom",
    },
}


def get_molecule_config(name: str):
    """Return the molecule configuration dict."""
    try:
        return MOLECULES[name]
    except KeyError:
        raise KeyError(
            f"Unknown molecule '{name}'. Available = {list(MOLECULES.keys())}"
        )
