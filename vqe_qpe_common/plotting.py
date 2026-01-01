"""
vqe_qpe_common.plotting
=======================

Centralised plotting utilities for the entire VQE/QPE package.

All PNG outputs are routed to:
    images/vqe/<MOLECULE>/   for VQE plots
    images/qpe/<MOLECULE>/   for QPE plots
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_ROOT = os.path.join(BASE_DIR, "images")


def format_token(val: object) -> str:
    if val is None:
        return ""
    if isinstance(val, (int, float)):
        s = f"{float(val):.5f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")
    s = str(val).strip()
    return s.replace(" ", "_").replace("+", "plus")


def format_molecule_name(mol: str) -> str:
    mol = str(mol).strip()
    mol = mol.replace("+", "plus")
    mol = mol.replace(" ", "_")
    return mol


def _fmt_noise_pct(p: float) -> str:
    return f"{int(round(float(p) * 100)):02d}"


def _fmt_float_token(x: float) -> str:
    s = f"{float(x):.6f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def _noise_tokens(
    *,
    dep: Optional[float],
    amp: Optional[float],
    noise_scan: bool,
    noise_type: Optional[str],
) -> list[str]:
    if noise_scan:
        nt = (noise_type or "").strip().lower()
        if nt in {"depolarizing", "dep"}:
            suffix = "dep"
        elif nt in {"amplitude", "amp", "amplitude_damping"}:
            suffix = "amp"
        elif nt in {"combined", "both"}:
            suffix = "combined"
        else:
            raise ValueError(
                "noise_scan=True requires noise_type in "
                "{depolarizing, amplitude, combined} "
                f"(got {noise_type!r})"
            )
        return [f"noise_scan_{suffix}"]

    toks: list[str] = []
    dep_f = float(dep or 0.0)
    amp_f = float(amp or 0.0)

    if dep_f > 0:
        toks.append(f"dep{_fmt_noise_pct(dep_f)}")
    if amp_f > 0:
        toks.append(f"amp{_fmt_noise_pct(amp_f)}")
    return toks


def build_filename(
    *,
    topic: str,
    ansatz: Optional[str] = None,
    optimizer: Optional[str] = None,
    mapping: Optional[str] = None,
    seed: Optional[int] = None,
    dep: Optional[float] = None,
    amp: Optional[float] = None,
    noise_scan: bool = False,
    noise_type: Optional[str] = None,
    multi_seed: bool = False,
    ancilla: Optional[int] = None,
    t: Optional[float] = None,
    tag: Optional[str] = None,
) -> str:
    topic = str(topic).strip().lower().replace(" ", "_")
    parts: list[str] = [topic]

    def _tok(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip().replace(" ", "_")
        return s if s else None

    for tkn in (_tok(ansatz), _tok(optimizer), _tok(mapping)):
        if tkn:
            parts.append(tkn)

    parts.extend(
        _noise_tokens(dep=dep, amp=amp, noise_scan=noise_scan, noise_type=noise_type)
    )

    if ancilla is not None:
        parts.append(f"ancilla{int(ancilla)}")

    if t is not None:
        parts.append(f"t{format_token(float(t))}")

    tg = _tok(tag)
    if tg:
        parts.append(tg)

    if (seed is not None) and (not multi_seed):
        parts.append(f"s{int(seed)}")

    return "_".join(parts) + ".png"


def _kind_dir(kind: str) -> str:
    k = str(kind).strip().lower()
    if k not in {"vqe", "qpe"}:
        raise ValueError(f"kind must be 'vqe' or 'qpe' (got {kind!r})")
    return os.path.join(IMG_ROOT, k)


def ensure_plot_dirs(*, kind: str, molecule: Optional[str] = None) -> str:
    target = _kind_dir(kind)
    if molecule:
        target = os.path.join(target, format_molecule_name(molecule))
    os.makedirs(target, exist_ok=True)
    return target


def save_plot(
    filename: str,
    *,
    kind: str,
    molecule: Optional[str] = None,
    show: bool = True,
) -> str:
    target_dir = ensure_plot_dirs(kind=kind, molecule=molecule)

    if not filename.lower().endswith(".png"):
        filename = filename + ".png"

    path = os.path.join(target_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"📁 Saved plot → {path}")
    return path
