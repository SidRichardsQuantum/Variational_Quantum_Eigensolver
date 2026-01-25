"""
tests.test_paths_ascii
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from common.plotting import (
    build_filename,
    ensure_plot_dirs,
    format_molecule_title,
)
from common.naming import format_molecule_name


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _assert_ascii_path_str(path: str) -> None:
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        assert _is_ascii(part), f"Non-ASCII path component: {part!r} in {path!r}"
        assert (
            "₂" not in part
        ), f"Subscript leaked into path component: {part!r} in {path!r}"


def _assert_ascii_path(p: Path) -> None:
    for part in p.parts:
        assert _is_ascii(part), f"Non-ASCII path component: {part!r} in {p}"
        assert "₂" not in part, f"Subscript leaked into path component: {part!r} in {p}"


def test_format_molecule_name_is_ascii_and_not_subscripted() -> None:
    out = format_molecule_name("H2")
    assert out == "H2"
    assert _is_ascii(out)
    assert "₂" not in out


def test_format_molecule_title_uses_subscripts_for_plot_titles_only() -> None:
    title = format_molecule_title("H2")
    assert "$_" in title
    assert "₂" not in title


@pytest.mark.parametrize("kind", ["vqe", "qpe", "qite"])
def test_ensure_plot_dirs_produces_ascii_paths(kind: str) -> None:
    d = ensure_plot_dirs(kind=kind, molecule="H2")
    _assert_ascii_path_str(d)


def test_build_filename_is_ascii_and_not_subscripted() -> None:
    fn = build_filename(topic="distribution", ancilla=4, t=1.0, dep=0.1, seed=0)
    assert fn.endswith(".png")
    assert _is_ascii(fn)
    assert "₂" not in fn


def test_results_dirs_are_ascii() -> None:
    import qite.io_utils as qite_io
    import qpe.io_utils as qpe_io
    import vqe.io_utils as vqe_io

    _assert_ascii_path(qite_io.RESULTS_DIR)
    _assert_ascii_path(qpe_io.RESULTS_DIR)
    _assert_ascii_path(vqe_io.RESULTS_DIR)


def test_vqe_prefix_and_signature_are_ascii() -> None:
    import vqe.io_utils as vqe_io

    cfg = {
        "molecule": "H2",
        "ansatz": "UCCSD",
        "optimizer": {"name": "Adam"},
        "depolarizing_prob": 0.1,
        "amplitude_damping_prob": 0.0,
    }
    h = vqe_io.run_signature(cfg)
    assert isinstance(h, str) and len(h) == 12
    assert _is_ascii(h)

    prefix = vqe_io.make_filename_prefix(
        cfg, noisy=True, seed=0, hash_str=h, algo="vqe"
    )
    assert _is_ascii(prefix)
    assert "₂" not in prefix


def test_qite_prefix_and_signature_are_ascii() -> None:
    import qite.io_utils as qite_io

    cfg = qite_io.make_run_config_dict(
        symbols=["H", "H"],
        coordinates=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        basis="sto-3g",
        seed=0,
        mapping="jordan_wigner",
        noisy=False,
        depolarizing_prob=0.0,
        amplitude_damping_prob=0.0,
        dtau=0.1,
        steps=10,
        molecule_label="H2",
        ansatz_name="UCCSD",
    )

    h = qite_io.run_signature(cfg)
    assert isinstance(h, str) and len(h) == 12
    assert _is_ascii(h)

    prefix = qite_io.make_filename_prefix(
        cfg, noisy=False, seed=0, hash_str=h, algo="qite"
    )
    assert _is_ascii(prefix)
    assert "₂" not in prefix


def test_qpe_cache_path_is_ascii() -> None:
    import qpe.io_utils as qpe_io

    key = qpe_io.signature_hash(
        molecule="H2",
        n_ancilla=4,
        t=1.0,
        seed=0,
        shots=1000,
        noise={"p_dep": 0.1, "p_amp": 0.0},
        trotter_steps=2,
        mapping="jordan_wigner",
        unit="angstrom",
    )
    assert isinstance(key, str) and len(key) == 12
    assert _is_ascii(key)

    p = qpe_io.cache_path(
        molecule="H2",
        n_ancilla=4,
        t=1.0,
        seed=0,
        noise={"p_dep": 0.1, "p_amp": 0.0},
        key=key,
        mapping="jordan_wigner",
        unit="angstrom",
    )
    _assert_ascii_path(p)
    assert p.suffix == ".json"


def test_existing_results_and_images_trees_are_ascii_if_present() -> None:
    """
    Final guardrail: if results/ or images/ exist in the working tree,
    all path components must be ASCII-only and contain no subscripts.
    """
    repo_root = Path(__file__).resolve().parent.parent

    for top in (repo_root / "results", repo_root / "images"):
        if not top.exists():
            continue

        for p in top.rglob("*"):
            _assert_ascii_path(p)
