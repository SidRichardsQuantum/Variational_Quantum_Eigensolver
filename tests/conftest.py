from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Reuse Matplotlib's font cache across tests and subprocesses. A per-test cache
# makes every CLI subprocess pay the several-second font-discovery cost again.
_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "vqe-pennylane-mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VQE_TEST_MODE", "1")
    monkeypatch.setenv("QPE_TEST_MODE", "1")
    monkeypatch.setenv("VQE_PENNYLANE_DATA_DIR", str(tmp_path))
    yield
