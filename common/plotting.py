"""
common.plotting
=======================

Centralised plotting utilities for the entire VQE/QPE package.

All PNG outputs are routed to:
    images/vqe/<MOLECULE>/   for VQE plots
    images/qpe/<MOLECULE>/   for QPE plots
"""

from __future__ import annotations

import os
import re
from typing import Optional

import matplotlib.pyplot as plt

from common.naming import format_token
from common.paths import images_dir

_SUB_RE = re.compile(r"([A-Za-z])(\d+)")


def slug_token(val: object) -> str:
    s = format_token(val)
    s = s.lower()
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def format_molecule_title(mol: str) -> str:
    """
    Title-safe molecule label (Matplotlib mathtext):

    - Subscript digits that immediately follow an element symbol: H2 -> H$_2$
    - Preserve non-filename-safe readability in titles only
    - Render trailing charge as a superscript: H3+ -> H$_3$$^{+}$ (as one math segment)

    Notes
    -----
    We avoid inserting raw '$...$' fragments before running the subscript regex,
    because that can create invalid nested mathtext.
    """
    s = str(mol).strip()

    # Extract a *trailing* charge like '+', '-', '++', '--'
    charge = ""
    m = re.search(r"([+-]+)$", s)
    if m:
        charge = m.group(1)
        s = s[: -len(charge)].strip()

    # Element + digits -> subscript (H2O -> H$_2$O)
    s = _SUB_RE.sub(r"\1$_\2$", s)

    # Apply charge as superscript at the end (mathtext)
    if charge:
        # normalize: '++' stays '++', etc.
        s = f"{s}$^{{{charge}}}$"

    return s


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
        parts.append(f"{int(ancilla)}ancilla")

    if t is not None:
        parts.append(f"t{_fmt_float_token(float(t))}")

    tg = _tok(tag)
    if tg:
        parts.append(tg)

    if (seed is not None) and (not multi_seed):
        parts.append(f"s{int(seed)}")

    return "_".join(parts) + ".png"


def ensure_plot_dirs(*, kind: str, molecule: Optional[str] = None) -> str:
    target = images_dir(kind, molecule=molecule)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


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
