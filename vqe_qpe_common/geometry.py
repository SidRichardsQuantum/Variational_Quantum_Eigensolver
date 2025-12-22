"""
vqe_qpe_common.geometry
===============

Shared geometry generation for bond scans, angle scans, etc.
Used identically by VQE and QPE.
"""

from __future__ import annotations

import numpy as np


def generate_geometry(name: str, param: float):
    """
    Geometry wrapper.
    Supported conventions (matching VQE):
        "H2_BOND"
        "H3+_BOND"
        "LiH_BOND"
        "H2O_ANGLE"
    """
    if name == "H2_BOND":
        return ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, param]])

    if name == "H3+_BOND":
        # Example equilateral-ish geometry
        return ["H", "H", "H"], np.array(
            [
                [0.0, 0.0, 0.0],
                [param, 0.0, 0.0],
                [0.5 * param, 0.866 * param, 0.0],
            ]
        )

    if name == "LiH_BOND":
        return ["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, param]])

    if name == "H2O_ANGLE":
        # Angle given in degrees
        theta = np.deg2rad(param)
        bond = 0.958
        return ["O", "H", "H"], np.array(
            [
                [0.0, 0.0, 0.0],
                [bond, 0.0, 0.0],
                [bond * np.cos(theta), bond * np.sin(theta), 0.0],
            ]
        )

    raise KeyError(f"Unknown geometry type: {name}")
