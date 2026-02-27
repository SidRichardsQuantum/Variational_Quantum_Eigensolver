"""
common/naming.py
"""

from __future__ import annotations


def format_molecule_name(mol: str) -> str:
    s = str(mol).strip()
    s = s.replace("+", "plus").replace("-", "minus").replace(" ", "_")
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def format_token(val: object) -> str:
    """
    Canonical filename-safe token.

    - ints/floats: fixed precision, trim trailing zeros, encode:
        '.' -> 'p', '-' -> 'm'
      Example: -1.25 -> 'm1p25'
    - strings: strip, spaces -> '_', '+' -> 'plus'
    """
    if val is None:
        return ""

    if isinstance(val, bool):
        return "true" if val else "false"

    if isinstance(val, int):
        return str(val)

    if isinstance(val, float):
        # Use 6 dp as a stable default for filenames; callers can pre-round if needed.
        s = f"{val:.6f}".rstrip("0").rstrip(".")
        s = s.replace("-", "m").replace(".", "p")
        return s

    s = str(val).strip()
    s = s.replace(" ", "_").replace("+", "plus").replace("-", "minus")
    return s
