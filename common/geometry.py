"""
common.geometry
===============

Shared geometry generation for bond scans, angle scans, etc.
Used identically by VQE and QPE.
"""

from __future__ import annotations

import numpy as np


def generate_geometry(name: str, param: float):
    """
    Geometry wrapper.
    Supported conventions:
        "H2_BOND"
        "H3+_BOND" (aliases: "H3PLUS_BOND", "H3_PLUS_BOND")
        "LiH_BOND"
        "H2O_ANGLE"
    """
    s = str(name).strip()
    if not s:
        raise KeyError("Unknown geometry type: empty name")

    up = s.upper().replace(" ", "").replace("-", "_")

    if up == "H2_BOND":
        p = float(param)
        return ["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, p]])

    if up in {"H3+_BOND", "H3PLUS_BOND", "H3_PLUS_BOND"}:
        p = float(param)
        # Example equilateral-ish geometry
        return ["H", "H", "H"], np.array(
            [
                [0.0, 0.0, 0.0],
                [p, 0.0, 0.0],
                [0.5 * p, 0.866 * p, 0.0],
            ],
            dtype=float,
        )

    if up == "LIH_BOND":
        p = float(param)
        return ["Li", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, p]])

    if up == "H2O_ANGLE":
        # Angle given in degrees
        theta = np.deg2rad(float(param))
        bond = 0.958
        return ["O", "H", "H"], np.array(
            [
                [0.0, 0.0, 0.0],
                [bond, 0.0, 0.0],
                [bond * np.cos(theta), bond * np.sin(theta), 0.0],
            ],
            dtype=float,
        )

    raise KeyError(f"Unknown geometry type: {name}")
