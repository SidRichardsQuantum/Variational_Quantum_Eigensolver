from __future__ import annotations

from pathlib import Path

import pytest

from common.naming import format_molecule_name
from common.plotting import build_filename, ensure_plot_dirs, format_molecule_title


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _assert_ascii_path(p: Path) -> None:
    for part in p.parts:
        assert _is_ascii(part), f"Non-ASCII path component: {part!r} in {p}"
        assert "₂" not in part, f"Subscript leaked into path component: {part!r} in {p}"


def test_format_molecule_name_is_ascii() -> None:
    out = format_molecule_name("H2")
    assert out == "H2"
    assert _is_ascii(out)
    assert "₂" not in out


def test_format_molecule_title_is_for_display_only() -> None:
    title = format_molecule_title("H2")
    assert "$_" in title
    assert "₂" not in title


@pytest.mark.parametrize("kind", ["vqe", "qpe", "qite"])
def test_ensure_plot_dirs_produces_ascii_paths(kind: str) -> None:
    d = Path(ensure_plot_dirs(kind=kind, molecule="H2"))
    _assert_ascii_path(d)


def test_build_filename_is_ascii() -> None:
    fn = build_filename(topic="distribution", ancilla=4, t=1.0, dep=0.1, seed=0)
    assert fn.endswith(".png")
    assert _is_ascii(fn)
    assert "₂" not in fn


def test_existing_results_and_images_paths_are_ascii_if_present() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    for top in (repo_root / "results", repo_root / "images"):
        if not top.exists():
            continue
        for p in top.rglob("*"):
            _assert_ascii_path(p)
