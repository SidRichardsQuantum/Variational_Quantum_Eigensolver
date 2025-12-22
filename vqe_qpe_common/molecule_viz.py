"""
vqe_qpe_common.molecule_viz
===========================

Minimal molecule diagram utilities (matplotlib-only):
- atoms (optionally with per-atom charge labels)
- inferred bonds + bond lengths
- inferred angles + bond angles

Designed for small molecules used in this repo (H2, LiH, H2O, H3+).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_COVALENT_RADII: Dict[str, float] = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "O": 0.66,
}

_ELEMENT_COLORS: Dict[str, str] = {
    "H": "#2A5ACB",
    "He": "#22D846",
    "Li": "#F080FF",
    "Be": "#E3E324",
    "O": "#FFFFFF",
}


def _as_xyz(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords, dtype=float)
    if c.ndim != 2 or c.shape[1] != 3:
        raise ValueError(f"coords must be shape (N,3); got {c.shape}")
    return c


def _project_to_2d(coords: np.ndarray) -> np.ndarray:
    """
    Project 3D coordinates to 2D using SVD/PCA on centered data.
    Returns array of shape (N, 2).
    """
    X = _as_xyz(coords)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Vt[:2].T
    return Xc @ P


def infer_bonds(
    symbols: Sequence[str],
    coords: np.ndarray,
    *,
    scale: float = 1.2,
    max_dist: float = 2.4,
) -> List[Tuple[int, int]]:
    """
    Bond if d(i,j) < scale*(r_i+r_j), capped by max_dist.
    """
    xyz = _as_xyz(coords)
    n = len(symbols)
    bonds: List[Tuple[int, int]] = []

    for i in range(n):
        ri = _COVALENT_RADII.get(symbols[i], 0.8)
        for j in range(i + 1, n):
            rj = _COVALENT_RADII.get(symbols[j], 0.8)
            d = float(np.linalg.norm(xyz[i] - xyz[j]))
            if d <= min(scale * (ri + rj), max_dist):
                bonds.append((i, j))
    return bonds


def infer_angles_from_bonds(
    bonds: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int, int]]:
    """
    Return angle triplets (i, j, k) where j is central and (i-j), (j-k) are bonds.
    """
    nbrs: Dict[int, List[int]] = {}
    for a, b in bonds:
        nbrs.setdefault(a, []).append(b)
        nbrs.setdefault(b, []).append(a)

    angles: List[Tuple[int, int, int]] = []
    for j, neigh in nbrs.items():
        if len(neigh) < 2:
            continue
        neigh = sorted(neigh)
        for u in range(len(neigh)):
            for v in range(u + 1, len(neigh)):
                i, k = neigh[u], neigh[v]
                angles.append((i, j, k))
    return angles


def bond_length(coords: np.ndarray, i: int, j: int) -> float:
    xyz = _as_xyz(coords)
    return float(np.linalg.norm(xyz[i] - xyz[j]))


def bond_angle_deg(coords: np.ndarray, i: int, j: int, k: int) -> float:
    xyz = _as_xyz(coords)
    v1 = xyz[i] - xyz[j]
    v2 = xyz[k] - xyz[j]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def plot_molecule(
    symbols: Sequence[str],
    coords: np.ndarray,
    *,
    title: Optional[str] = None,
    bonds: Optional[Sequence[Tuple[int, int]]] = None,
    angles: Optional[Sequence[Tuple[int, int, int]]] = None,
    atom_charges: Optional[Sequence[float]] = None,
    show_bond_lengths: bool = True,
    show_angles: bool = True,
    show_atom_indices: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes:
    """
    Minimal 2D molecule diagram:
    - atoms as circles
    - bonds as lines + length labels
    - angles as text labels near the central atom
    - optional per-atom charge labels (useful for ions)
    """

    # -----------------------------------------------------------------
    # Backwards compatibility for noteboooks
    # -----------------------------------------------------------------
    if "show_bond_angles" in kwargs:
        show_angles = bool(kwargs.pop("show_bond_angles"))

    # If anything else was passed, fail loudly
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")

    xyz = _as_xyz(coords)
    n = len(symbols)
    if atom_charges is not None and len(atom_charges) != n:
        raise ValueError("atom_charges must have length N")

    xy = _project_to_2d(xyz)

    bonds_use = list(infer_bonds(symbols, xyz) if bonds is None else bonds)
    angles_use = list(infer_angles_from_bonds(bonds_use) if angles is None else angles)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Bonds
    for i, j in bonds_use:
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], linewidth=2.2, alpha=0.9)
        if show_bond_lengths:
            d = bond_length(xyz, i, j)
            mid = 0.5 * (xy[i] + xy[j])
            ax.text(mid[0], mid[1], f"{d:.3f} Å", fontsize=9, ha="center", va="center")

    # Atoms
    for idx, (sym, (x, y)) in enumerate(zip(symbols, xy)):
        col = _ELEMENT_COLORS.get(sym, "#BBBBBB")
        ax.scatter([x], [y], s=380, c=col, edgecolors="k", linewidths=0.9, zorder=3)

        label = sym
        if show_atom_indices:
            label += str(idx)
        if atom_charges is not None:
            label += f"\n{float(atom_charges[idx]):+.2f}"

        ax.text(x, y, label, fontsize=10, ha="center", va="center", zorder=4)

    # Angles (text near central atom j)
    if show_angles:
        for i, j, k in angles_use:
            ang = bond_angle_deg(xyz, i, j, k)
            if np.isnan(ang):
                continue
            v = (xy[i] - xy[j]) + (xy[k] - xy[j])
            nv = np.linalg.norm(v)
            v = v / nv if nv > 1e-12 else np.array([0.0, 1.0])
            pos = xy[j] + 0.25 * v
            ax.text(pos[0], pos[1], f"{ang:.1f}°", fontsize=9, ha="center", va="center")

    ax.set_title(title or "Molecule geometry")
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    return ax
