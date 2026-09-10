"""Runtime artifacts belong to user storage, independently of import location."""

import sys

import pennylane as qml
import pytest

from common.paths import data_root, images_dir, results_dir
from qite import run_qite, run_qrte
from qpe import run_qpe
from vqe import run_vqe


def test_default_data_root_is_user_storage(monkeypatch, tmp_path):
    monkeypatch.delenv("VQE_PENNYLANE_DATA_DIR")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    if sys.platform == "win32":
        expected = tmp_path / "AppData" / "Local"
    elif sys.platform == "darwin":
        expected = tmp_path / "Library" / "Application Support"
    else:
        expected = tmp_path / ".local" / "share"
    assert data_root() == expected / "vqe-pennylane"
    assert not data_root().exists()


@pytest.mark.skipif(sys.platform in {"win32", "darwin"}, reason="XDG applies on Linux")
def test_xdg_user_data_override(monkeypatch, tmp_path):
    monkeypatch.delenv("VQE_PENNYLANE_DATA_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert data_root() == tmp_path / "vqe-pennylane"


@pytest.mark.parametrize(
    "runner,options,kind",
    [
        (run_vqe, {"ansatz_name": "RY-CZ", "steps": 1, "reference_state": [1]}, "vqe"),
        (
            run_qite,
            {"ansatz_name": "RY-CZ", "steps": 1, "reference_state": [1]},
            "qite",
        ),
        (
            run_qrte,
            {"ansatz_name": "RY-CZ", "steps": 1, "reference_state": [1]},
            "qite",
        ),
        (run_qpe, {"n_ancilla": 2, "shots": None, "hf_state": [1]}, "qpe"),
    ],
)
def test_runtime_override_after_import_redirects_cache(
    monkeypatch, tmp_path, runner, options, kind
):
    # Imports above have already loaded the I/O modules before either override.
    for folder in [tmp_path / "first", tmp_path / "second"]:
        monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(folder))
        cfg = dict(
            hamiltonian=qml.Hamiltonian([1.0], [qml.Z(0)]), plot=False, **options
        )
        fresh = runner(**cfg)
        cached = runner(**cfg)
        assert fresh["cache_hit"] is False
        assert cached["cache_hit"] is True
        assert list((folder / "results" / kind).glob("*.json"))
        assert results_dir(kind) == folder / "results" / kind
        assert images_dir(kind) == folder / "images" / kind
